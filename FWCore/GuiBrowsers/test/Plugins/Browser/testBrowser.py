#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

import Path
from Vispa.Main.Directories import *

from Vispa.Main.Application import Application
from Vispa.Plugins.Browser.BrowserPlugin import BrowserPlugin
from Vispa.Plugins.Browser.BrowserTab import BrowserTab
from Vispa.Plugins.Browser.BrowserTabController import BrowserTabController
from TestDataAccessor import TestDataAccessor
from Vispa.Share import Profiling

class BrowserTestCase(unittest.TestCase):
    def testBrowser(self):
        logging.debug(self.__class__.__name__ +': testRun()')
        self.app=Application(sys.argv)
        self.app.mainWindow().setWindowTitle("test Browser")
        self.plugin=BrowserPlugin(self.app)
        self.plugin.startUp()
        self.tab = BrowserTab(self.app.mainWindow())
        self.controller = BrowserTabController(self.plugin)
        self.controller.setDataAccessor(TestDataAccessor())
        self.tab.setController(self.controller)
        self.app.mainWindow().addTab(self.tab)
        self.controller.updateContent()
        self.controller.updateViewMenu()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.run()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
