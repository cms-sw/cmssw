#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtCore import QCoreApplication
from PyQt4.QtGui import QApplication,QMainWindow

import Path
from Vispa.Main.Directories import *

from Vispa.Gui.ZoomableScrollArea import ZoomableScrollArea
from Vispa.Gui.VispaWidget import VispaWidget
from Vispa.Gui.ZoomableWidget import ZoomableWidget
from TestDataAccessor import TestDataAccessor
from Vispa.Share import Profiling

class ZoomableWidgetTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ + ': testExample()')
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("test ZoomableWidget")
        self.window.resize(300,300)
        self.app.setActiveWindow(self.window)
        self.window.show()
        self.scrollArea = ZoomableScrollArea(self.window)
        self.window.setCentralWidget(self.scrollArea)
        self.zoomableWidget = ZoomableWidget()
        self.scrollArea.setWidget(self.zoomableWidget)
        self.widget=VispaWidget(self.zoomableWidget)
        self.widget.move(10,10)
        self.widget.show()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__,"ZoomableScrollArea|VispaWidget|ZoomableWidget")
