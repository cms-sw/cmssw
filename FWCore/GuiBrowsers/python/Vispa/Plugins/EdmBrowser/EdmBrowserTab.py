import logging

from PyQt4.QtGui import QWidget

from Vispa.Gui.Header import FrameWithHeader
from Vispa.Plugins.Browser.BrowserTab import BrowserTab
from Vispa.Plugins.EdmBrowser.BranchTableView import BranchTableView

class EdmBrowserTab(BrowserTab):
    def __init__(self, parent):
        BrowserTab.__init__(self, parent)
    
    def createTreeView(self,parent=None):
        """ Create the tree view.
        """
        if not parent:
            parent=self.horizontalSplitter()
        
        self._treeviewArea = FrameWithHeader(parent)
        self._treeviewArea.header().setText("Branches")
        self._treeViewMenuButton = self._treeviewArea.header().createMenuButton()
        self._treeView = BranchTableView(self._treeviewArea)
        self._treeviewArea.addWidget(self._treeView)
