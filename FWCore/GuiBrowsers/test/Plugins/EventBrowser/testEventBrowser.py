#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

import Path
from Vispa.Main.Directories import *

from Vispa.Main.Application import Application
from Vispa.Plugins.EventBrowser.EventBrowserPlugin import EventBrowserPlugin
from Vispa.Plugins.Browser.BrowserTab import BrowserTab
from Vispa.Plugins.EventBrowser.EventBrowserTabController import EventBrowserTabController
from TestDataAccessor import TestDataAccessor
from Vispa.Share import Profiling

class EventBrowserTestCase(unittest.TestCase):
    def testEventBrowser(self):
        logging.debug(self.__class__.__name__ +': testRun()')
        self.app=Application(sys.argv)
        self.app.mainWindow().setWindowTitle("test EventBrowser")
        self.plugin=EventBrowserPlugin(self.app)
        self.plugin.startUp()
        self.tab = BrowserTab(self.app.mainWindow())
        self.controller = EventBrowserTabController(self.plugin)
        self.controller.setDataAccessor(TestDataAccessor())
        self.tab.setController(self.controller)
        self.app.mainWindow().addTab(self.tab)
        self.controller.updateContent()
        self.controller.updateViewMenu()
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.run()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
