#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

import Path
from Vispa.Main.Directories import *
examplesDirectory = os.path.join(baseDirectory,"examples/EdmBrowser")

from Vispa.Main.Application import *
from Vispa.Share import Profiling

class EdmBrowserTestCase(unittest.TestCase):
    def testEdmBrowser(self):
        logging.debug(self.__class__.__name__ +': testRun()')
        self.app=Application(sys.argv)
        self.app.mainWindow().setWindowTitle("test EdmBrowser")
        self.app.openFile(os.path.join(examplesDirectory,"QCDDiJet_Pt50to80_Summer09_RECO_3_1_X_10events.root"))
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.run()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
