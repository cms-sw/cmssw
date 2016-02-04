#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtCore import SIGNAL
from PyQt4.QtGui import QApplication,QMainWindow

import Path
from Vispa.Main.Directories import *
setBaseDirectory(baseDirectory)

from Vispa.Gui.BoxContentDialog import BoxContentDialog
from Vispa.Share import Profiling

class BoxContentDialogTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ +': testExample()')
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.app.setActiveWindow(self.window)
        self._boxContentDialog=BoxContentDialog(self.window)
        self._boxContentDialog.addButton("&label","str(object.label)")
        self.app.connect(self._boxContentDialog, SIGNAL("scriptChanged"), self.scriptChanged)
        self._boxContentDialog.onScreen()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()
        
    def scriptChanged(self,script):
        logging.debug(self.__class__.__name__ +': script changed '+str(script))

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__,"BoxContentDialog")
