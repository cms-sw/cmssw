#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

import Path
from Vispa.Main.Directories import *
examplesDirectory = os.path.join(baseDirectory,"examples/ConfigEditor")
if not os.path.exists(examplesDirectory):
    examplesDirectory = os.path.abspath(os.path.join(os.path.join(baseDirectory,".."),"examples"))

from Vispa.Main.Application import Application
from Vispa.Share import Profiling

class ConfigBrowserTestCase(unittest.TestCase):
    def testConfigBrowser(self):
        global test
        logging.debug(self.__class__.__name__ +': testRun()')
        self.app=Application(sys.argv)
        self.app.mainWindow().setWindowTitle("test ConfigBrowser")
        self.app.openFile(os.path.join(examplesDirectory,"patLayer1_fromAOD_full_cfg.py"))
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.run()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
