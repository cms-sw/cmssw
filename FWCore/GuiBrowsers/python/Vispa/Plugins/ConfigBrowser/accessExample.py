#! /usr/bin/env python
import unittest
import os.path
import sys

baseDirectory=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
sys.path.append(baseDirectory)

from Vispa.Plugins.ConfigBrowser.ConfigDataAccessor import *

accessor=ConfigDataAccessor()
accessor.open(os.path.join("../../../examples/Plugins/ConfigBrowser","patLayer1_fromLayer0_full_cfg_CMSSW_2_1_X.py"))
objects=accessor.topLevelObjects()
print "toplevelobjects: ",objects
objects2=accessor.children(objects[0])
print "children of first object: ",objects2
objects3=accessor.children(objects2[0])
print "children of first object: ",objects3
print "label: ",accessor.label(objects3[0])
print "properties: ",accessor.properties(objects3[0])
