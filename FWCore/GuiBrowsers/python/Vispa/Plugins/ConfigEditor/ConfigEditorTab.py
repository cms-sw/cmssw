import logging

from PyQt4.QtCore import SIGNAL, Qt
from PyQt4.QtGui import QWidget, QListWidget, QVBoxLayout, QPushButton,QToolButton, QSplitter

from Vispa.Main.SplitterTab import SplitterToolBar
from Vispa.Plugins.Browser.BrowserTab import BrowserTab
from Vispa.Gui.ToolBoxContainer import ToolBoxContainer
from Vispa.Plugins.ConfigEditor.CodeTableView import CodeTableView

class ConfigEditorTab(BrowserTab):
    """ This is the main frame of the Config Editor Plugin.
    
    The tab is split in three horizontal parts, from left to right: Tree View, Center View, Property View.
    The editor can be dispayed using createEditor(). The Property View is then shared with the ConfigBrowser.
    """
    
    def __init__(self, parent=None):
        """ constructor """
        logging.debug(self.__class__.__name__ +": __init__()")
        BrowserTab.__init__(self, parent,True)
        self._editorSplitter = None
    
    def createEditor(self):
        self.createToolBar(0)
        self._editorSplitter = QSplitter(Qt.Horizontal)

        toolBarSectionId = self.toolBar().addSection(SplitterToolBar.ALIGNMENT_LEFT)

        self._editorTableView=CodeTableView(self._editorSplitter)
        
        self._editorSplitter.setSizes([100, 300])

        self._minimizeButton = QToolButton()
        self._minimizeButton.setText("v")
        self._minimizeButton.setCheckable(True)
        self._originalButton = QToolButton()
        self._originalButton.setText("-")
        self._originalButton.setCheckable(True)
        self._originalButton.setChecked(True)
        self._maximizeButton = QToolButton()
        self._maximizeButton.setText("^")
        self._maximizeButton.setCheckable(True)

        self._centerViewToolBarId = self.toolBar().addSection(SplitterToolBar.ALIGNMENT_CENTER)
        self.toolBar().addWidgetToSection(self._minimizeButton, self._centerViewToolBarId)
        self.toolBar().addWidgetToSection(self._originalButton, self._centerViewToolBarId)
        self.toolBar().addWidgetToSection(self._maximizeButton, self._centerViewToolBarId)
    
        self.verticalSplitter().insertWidget(0,self._editorSplitter)
        self.updateToolBarSizes()
            
        controller = self.controller()
        if controller:
            self.connect(self._minimizeButton, SIGNAL('clicked(bool)'), controller.minimizeEditor)
            self.connect(self._originalButton, SIGNAL('clicked(bool)'), controller.originalEditor)
            self.connect(self._maximizeButton, SIGNAL('clicked(bool)'), controller.maximizeEditor)
        
    def editorSplitter(self):
        return self._editorSplitter
        
    def horizontalSplitterMovedSlot(self, pos, index):
        if self.toolBar(): 
            self.updateToolBarSizes()
        
    def updateToolBarSizes(self):
        self.toolBar().setSectionSizes(self.horizontalSplitter().sizes())

    def minimizeButton(self):
        return self._minimizeButton

    def originalButton(self):
        return self._originalButton

    def maximizeButton(self):
        return self._maximizeButton

    def editorTableView(self):
        return self._editorTableView
    