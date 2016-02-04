#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtGui import QApplication,QMainWindow

import Path
from Vispa.Main.Directories import *

from Vispa.Main.Exceptions import exception_traceback
from Vispa.Gui.ZoomableScrollArea import ZoomableScrollArea
from Vispa.Views.LineDecayView import LineDecayView
from TestDataAccessor import TestDataAccessor
from Vispa.Share import Profiling

class LineDecayViewTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ + ': testExample()')
        try:
            from pxl.algorithms import AutoLayout
        except Exception:
            logging.info("LineDecayView needs PXL: " + exception_traceback())
            return
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("test LineDecayView")
        self.window.resize(300,300)
        self.app.setActiveWindow(self.window)
        self.window.show()
        self.scrollArea = ZoomableScrollArea(self.window)
        self.window.setCentralWidget(self.scrollArea)
        self.lineDecayView = LineDecayView()
        self.scrollArea.setWidget(self.lineDecayView)
        accessor=TestDataAccessor()
        self.lineDecayView.setDataAccessor(accessor)
        self.lineDecayView.setDataObjects(accessor.topLevelObjects())
        self.lineDecayView.updateContent()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__,"LineDecayView")
