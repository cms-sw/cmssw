#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtGui import QApplication,QMainWindow

import Path
from Vispa.Main.Directories import *

from Vispa.Views.TreeView import TreeView
from TestDataAccessor import TestDataAccessor
from Vispa.Share import Profiling

class TreeViewTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ +': testExample()')
        self.app = QApplication(sys.argv)
        self.window= QMainWindow()
        self.window.setWindowTitle("test TreeView")
        self.app.setActiveWindow(self.window)
        self.window.show()
        self.treeView = TreeView(self.window)
        self.window.setCentralWidget(self.treeView)
        accessor=TestDataAccessor()
        self.treeView.setDataAccessor(accessor)
        self.treeView.setDataObjects(accessor.topLevelObjects())
        self.treeView.updateContent()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__,"TreeView")
