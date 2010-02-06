import logging

from PyQt4.QtGui import QWidget

from Vispa.Plugins.Browser.BrowserTab import BrowserTab
from Vispa.Plugins.EdmBrowser.BranchTableView import BranchTableView

class EdmBrowserTab(BrowserTab):
    def __init__(self, parent):
        BrowserTab.__init__(self, parent)
    
    def createTreeView(self):
        self._treeView=BranchTableView(self.horizontalSplitter())
        self._treeViewHeader=QWidget()
        self._treeViewMenuButton=QWidget()
        
