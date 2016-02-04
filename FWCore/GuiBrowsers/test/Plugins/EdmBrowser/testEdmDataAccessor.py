#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

import Path
from Vispa.Main.Directories import *
sys.path.append(os.path.join(baseDirectory,"Vispa/Plugins/EdmBrowser"))
examplesDirectory = os.path.join(baseDirectory,"examples/EdmBrowser")

from Vispa.Main.Exceptions import *
try:
    from EdmDataAccessor import *
except Exception:
    logging.error("Cannot open EdmDataAccessor: " + exception_traceback())
from Vispa.Share import Profiling

def countObjects(accessor,object=None,i=0):
    #logging.debug(__name__ + ": countObjects")
    i+=1
    if object==None:
        for child in accessor.topLevelObjects():
            i=countObjects(accessor,child,i)
    else:
        for child in accessor.children(object):
            i=countObjects(accessor,child,i)
    return i 

def countMotherRelations(accessor,object=None,i=0):
    i+=len(accessor.motherRelations(object))
    if object==None:
        for child in accessor.topLevelObjects():
            i=countMotherRelations(accessor,child,i)
    else:
        for child in accessor.children(object):
            i=countMotherRelations(accessor,child,i)
    return i 

def countDaughterRelations(accessor,object=None,i=0):
    i+=len(accessor.daughterRelations(object))
    if object==None:
        for child in accessor.topLevelObjects():
            i=countDaughterRelations(accessor,child,i)
    else:
        for child in accessor.children(object):
            i=countDaughterRelations(accessor,child,i)
    return i 

class EdmDataAccessorTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ +': testExample()')
        accessor=EdmDataAccessor()
        accessor.open(os.path.join(examplesDirectory,"QCDDiJet_Pt50to80_Summer09_RECO_3_1_X_10events.root"))
        
        self.assertEqual(accessor.numberOfEvents(),10)
        self.assertEqual(accessor.eventNumber(),1)
#        self.assertEqual(countMotherRelations(accessor),3)
#        self.assertEqual(countDaughterRelations(accessor),3)
#        self.assertEqual(int(float(accessor.propertyValue(accessor.children(accessor.children(accessor.topLevelObjects()[0])[0])[2],"Mass"))),92)

        accessor.next()
        self.assertEqual(accessor.eventNumber(),2)
#        self.assertEqual(int(float(accessor.propertyValue(accessor.children(accessor.children(accessor.topLevelObjects()[0])[0])[2],"Mass"))),96)

        accessor.first()
        self.assertEqual(accessor.eventNumber(),1)
#        self.assertEqual(int(float(accessor.propertyValue(accessor.children(accessor.children(accessor.topLevelObjects()[0])[0])[2],"Mass"))),92)

        accessor.last()
        self.assertEqual(accessor.eventNumber(),10)
#        self.assertEqual(int(float(accessor.propertyValue(accessor.children(accessor.children(accessor.topLevelObjects()[0])[0])[2],"Mass"))),91)

        accessor.previous()
        self.assertEqual(accessor.eventNumber(),9)
#        self.assertEqual(int(float(accessor.propertyValue(accessor.children(accessor.children(accessor.topLevelObjects()[0])[0])[2],"Mass"))),90)

        accessor.goto(8)
        self.assertEqual(accessor.eventNumber(),8)
#        self.assertEqual(int(float(accessor.propertyValue(accessor.children(accessor.children(accessor.topLevelObjects()[0])[0])[2],"Mass"))),89)

        self.assertEqual(accessor.numberOfEvents(),10)

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
