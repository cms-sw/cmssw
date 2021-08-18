#!/usr/bin/env python3
from __future__ import print_function
import keras
import theano

from keras.models import Sequential
from keras.layers import Dense

print(keras.__version__)
print(theano.__version__)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
