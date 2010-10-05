#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtGui import QApplication,QMainWindow

import Path
from Vispa.Main.Directories import *

from Vispa.Gui.ZoomableScrollArea import ZoomableScrollArea
from Vispa.Views.BoxDecayView import BoxDecayView
from TestDataAccessor import TestDataAccessor
from Vispa.Share import Profiling

class BoxDecayViewTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ + ': testExample()')
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("test BoxDecayView")
        self.window.resize(300,300)
        self.app.setActiveWindow(self.window)
        self.window.show()
        self.scrollArea = ZoomableScrollArea(self.window)
        self.window.setCentralWidget(self.scrollArea)
        self.boxDecayView = BoxDecayView()
        self.scrollArea.setWidget(self.boxDecayView)
        accessor=TestDataAccessor()
        self.boxDecayView.setDataAccessor(accessor)
        self.boxDecayView.setDataObjects(accessor.topLevelObjects())
        self.boxDecayView.setBoxContentScript("str(object.Label)")
        self.boxDecayView.updateContent()
        for w in self.boxDecayView.children():
            if hasattr(w, "setDragable"):
                w.setDragable(True, True)

        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
