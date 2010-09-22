#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtGui import QApplication,QMainWindow

import Path
from Vispa.Main.Directories import *

from Vispa.Views.PropertyView import PropertyView
from TestDataAccessor import TestDataAccessor
from Vispa.Share import Profiling

class PropertyViewTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ +': testExample()')
        self.app = QApplication(sys.argv)
        self.window= QMainWindow()
        self.window.setWindowTitle("test PropertyView")
        self.app.setActiveWindow(self.window)
        self.window.show()
        self.propertyView=PropertyView(self.window)
        self.window.setCentralWidget(self.propertyView)
        self.propertyView.setDataAccessor(TestDataAccessor())
        self.propertyView.setDataObject("particle1")
        self.propertyView.updateContent()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__,"PropertyView")
