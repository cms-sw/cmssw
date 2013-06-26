#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtGui import QApplication,QMainWindow,QTextEdit

import Path

from Vispa.Gui.ToolBoxContainer import ToolBoxContainer
from Vispa.Share import Profiling

class ToolBoxContainerTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ +': testExample()')
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("test ToolBoxContainer")
        self.app.setActiveWindow(self.window)
        self.window.show()
        container = ToolBoxContainer(self.window)
        self.window.setCentralWidget(container)
        container.addWidget(QTextEdit("ganz viel text\n mit zeilenumbruechen\n."))
        container.addWidget(QTextEdit("anderer inhalt."))
        container.show()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__,"ToolBoxContainer")
