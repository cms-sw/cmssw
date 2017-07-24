# -*- coding: utf-8 -*-

"""
Test script to create a simple graph for testing purposes at bin/data.
"""


import os
import tensorflow as tf


thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

x_ = tf.placeholder(tf.float32, [None, 10], name="input")

W = tf.Variable(tf.ones([10, 1]))
b = tf.Variable(tf.ones([1]))
y = tf.add(tf.matmul(x_, W), b, name="output")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={x_: [range(10)]})[0][0])

saver = tf.train.Saver()
saver.save(sess, os.path.join(datadir, "simplegraph"))
