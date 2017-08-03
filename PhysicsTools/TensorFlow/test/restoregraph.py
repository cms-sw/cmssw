# -*- coding: utf-8 -*-

"""
Simple test to restore and evaluate the graph produced by creategraph.py and saved at bin/data/
using the SavedModel serialization format.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
"""


import os
import tensorflow as tf


thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

sess = tf.Session()

tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                           os.path.join(datadir, "simplegraph"))

print(sess.run({"y": "output:0"}, feed_dict={"input:0": [range(10)]})["y"][0][0])
