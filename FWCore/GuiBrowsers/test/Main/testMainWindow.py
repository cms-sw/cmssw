#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

import Path
from Vispa.Main.Directories import *

from Vispa.Main.Application import *
from Vispa.Share import Profiling

class MainWindowTestCase(unittest.TestCase):
    def testRun(self):
        logging.debug(self.__class__.__name__ +': testRun()')
        self.app=Application(sys.argv)
        self.app.mainWindow().setWindowTitle("test MainWindow")
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.run()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
