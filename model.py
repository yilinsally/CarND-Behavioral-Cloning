import csv
import numpy as np
import cv2
import os
import sklearn
import random


#import driving_log csv file to obtain images and measurements
lines = []
with open('./data3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
			lines.append(line)

#split the training and validation data 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)




# define a generator and set the batch size equal to 32
def generator(lines, batch_size=32):
	num_of_lines = len(lines)
	while 1:
		for offset in range(0, num_of_lines, batch_size):
			batch_samples = lines[offset:offset+batch_size]

			images = []
			measurements = []
			for line in batch_samples:
				for i in range(3):
					current_path = line[i]
					filename = current_path.split('/')[-1]
					path = './data3/IMG/'+filename
					image = cv2.imread(path)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					images.append(image)
				correction = 0.2
				measurement = float(line[3])
				left_steering = measurement + correction
				right_steering = measurement - correction
				measurements.extend([measurement,left_steering,right_steering])
			#augmented_images = []
			#augmented_measurements = []
			#for image,measurment in zip (images, measurements):
				#num = np.random.randint(0,1)
				#if num == 0:
					#flipped = cv2.flip(image, 1)
					#flipped_measurement = measurement *(-1)
					#augmented_images.append(flipped)
					#augmented_measurements.append(flipped_measurement)

				#else:
					#augmented_images.append(image)
					#augmented_measurements.append(measurement)
 
	
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

#define training generator and validation generator
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)




from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


#build a convolutional neural network
model = Sequential()

model.add(Lambda(lambda x: x/127.5-1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping =((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Dropout(0.5))

model.add(Dense(1))


model.compile(loss='mse', optimizer= 'adam')

history = model.fit_generator(train_generator, steps_per_epoch = len(train_samples)/32, validation_data = validation_generator, validation_steps = len(validation_samples)/32 , epochs = 10)

model.save('model.h5')



















