#!/usr/bin/env python
import keras
import theano
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D

print keras.__version__
print theano.__version__

model = Sequential()
model.add(Conv2D(32, 10, 10,border_mode='valid',input_shape = (1,100,100)))
model.add(MaxPooling2D(pool_size=(2, 2)))
