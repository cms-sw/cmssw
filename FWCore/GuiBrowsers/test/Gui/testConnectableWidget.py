#! /usr/bin/env python
import unittest
import os.path
import sys

import logging
logging.root.setLevel(logging.DEBUG)

from PyQt4.QtGui import QApplication,QMainWindow

import Path
from Vispa.Main.Directories import *

from Vispa.Gui.ConnectableWidget import ConnectableWidget
from Vispa.Share import Profiling

class ConnectableWidgetTestCase(unittest.TestCase):
    def testExample(self):
        logging.debug(self.__class__.__name__ +': testExample()')
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("test ConnectableWidget")
        self.app.setActiveWindow(self.window)
        self.window.resize(800, 600)
        self.window.show()
        
        widgets = []
        
        rowOneY = 10
        
        widget = ConnectableWidget(self.window, "empty")
        widget.move(10, rowOneY)
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window, "with ports")
        widget.move(210, rowOneY)
        widget.addSinkPort("first sink port")
        widget.addSourcePort("first source port")
        widget.addSourcePort("second sourcePort")
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window, "with port names")
        widget.move(410, rowOneY)
        widget.setShowPortNames(True)
        widget.addSinkPort("first sink port")
        widget.addSourcePort("first source port")
        widget.addSourcePort("second sourcePort")
        widgets.append(widget)
        
        # ________________________________ autocale _______________________________
        rowTwoY = 130
                
        widget = ConnectableWidget(self.window, "autocale")
        widget.move(10, rowTwoY)
        widget.enableAutosizing(True, False)
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window, "with ports")
        widget.move(210, rowTwoY)
        widget.enableAutosizing(True, False)
        widget.addSinkPort("first sink port")
        widget.addSourcePort("first source port")
        widget.addSourcePort("second sourcePort")
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window, "with port names sdfsdfg dfgsdf")
        widget.move(410, rowTwoY)
        widget.enableAutosizing(True, False)
        widget.setShowPortNames(True)
        widget.addSinkPort("first sink port")
        widget.addSourcePort("first source port")
        widget.addSourcePort("second sourcePort")
        widgets.append(widget)
        
        # ________________________ special _____________________
        rowThreeX = 230
        
        widget = ConnectableWidget(self.window)
        widget.move(10, rowThreeX)
        widget.enableAutosizing(True, False)
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window)
        widget.move(80, rowThreeX)
        widget.enableAutosizing(True, False)
        widget.addSinkPort("first sink port")
        widget.addSourcePort("first source port")
        widget.addSourcePort("second sourcePort")
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window)
        widget.move(150, rowThreeX)
        widget.enableAutosizing(True, False)
        widget.setText("with text")
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window)
        widget.move(220, rowThreeX)
        widget.enableAutosizing(True, False)
        widget.setTitle("with title")
        widget.setText("with text")
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window, "many ports")
        widget.move(410, rowThreeX)
        widget.enableAutosizing(True, False)
        widget.setShowPortNames(True)
        widget.addSinkPort("first sink port")
        widget.addSinkPort("another sink port")
        widget.addSinkPort("another sink port")
        widget.addSinkPort("another sink port")
        widget.addSinkPort("another sink port")
        widget.addSourcePort("first source port")
        widget.addSourcePort("second sourcePort")
        widget.addSourcePort("another sourcePort")
        widget.addSourcePort("another sourcePort")
        widget.addSourcePort("another sourcePort")
        widget.addSourcePort("another sourcePort")
        widget.addSourcePort("another sourcePort")
        widget.addSourcePort("another sourcePort")
        widget.addSourcePort("another sourcePort")
        widget.addSourcePort("another sourcePort")
        widget.addSourcePort("another sourcePort")
        widgets.append(widget)
        
        # _________________________ ports and text __________________________
        rowFourX = 300
        
        
        widget = ConnectableWidget(self.window)
        widget.move(10, rowFourX)
        widget.enableAutosizing(True, False)
        widget.setTitle("with title")
        widget.setText("with text")
        widget.addSinkPort("first sink port")
        widget.addSourcePort("first source port")
        widget.addSourcePort("second sourcePort")
        widgets.append(widget)
        
        widget = ConnectableWidget(self.window)
        widget.move(120, rowFourX)
        widget.enableAutosizing(True, False)
        widget.setShowPortNames(True)
        widget.setTitle("with port names and text")
        widget.setText("with text")
        widget.addSinkPort("first sink port")
        widget.addSourcePort("first source port")
        widget.addSourcePort("second sourcePort")
        widgets.append(widget)
        
        # _________________________ test __________________________
        rowFiveX = 370
        
        
        
        for widget in widgets:
            widget.scheduleRearangeContent()
            widget.show()
        
        if not hasattr(unittest,"NO_GUI_TEST"):
            self.app.exec_()

if __name__ == "__main__":
    Profiling.analyze("unittest.main()",__file__)
