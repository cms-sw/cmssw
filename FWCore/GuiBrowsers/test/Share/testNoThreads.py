#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

import Path

from Vispa.Main.Application import Application
Application.NO_PROCESS_EVENTS=True
from Vispa.Share.ThreadChain import ThreadChain
ThreadChain.NO_THREADS_FLAG=True

from Vispa.Share import Profiling

class MainWindowTestCase(unittest.TestCase):
    def testRun(self):
        logging.debug(self.__class__.__name__ +': testRun()')
        self.app=Application(sys.argv)
        self.app.mainWindow().setWindowTitle("test no threads")
        self.app.run()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
