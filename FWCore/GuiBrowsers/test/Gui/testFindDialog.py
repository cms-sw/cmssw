#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtCore import SIGNAL
from PyQt4.QtGui import QApplication,QMainWindow

import Path

from TestDataAccessor import TestDataAccessor
from Vispa.Gui.FindDialog import FindDialog
from Vispa.Share.FindAlgorithm import FindAlgorithm
from Vispa.Share import Profiling

class FindDialogTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ +': testExample()')
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.app.setActiveWindow(self.window)
        self._findAlgoritm=FindAlgorithm()
        accessor=TestDataAccessor()
        self._findAlgoritm.setDataAccessor(accessor)
        self._findAlgoritm.setDataObjects(accessor.topLevelObjects())
        self._findDialog=FindDialog(self.window)
        self._findDialog.setFindAlgorithm(self._findAlgoritm)
        self._findDialog.setLabel("particle")
        self.app.connect(self._findDialog, SIGNAL("found"), self.found)
        self._found=False
        self._findDialog.findNext()
        self.assertEqual(self._found,True)
        self._findDialog.onScreen()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()
        
    def found(self,object):
        logging.debug(self.__class__.__name__ +': found '+str(object))
        self._found=object=="particle1"

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__,"FindDialog|FindAlgorithm")
