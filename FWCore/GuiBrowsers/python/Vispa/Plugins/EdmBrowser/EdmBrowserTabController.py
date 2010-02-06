import logging

from PyQt4.QtCore import SIGNAL,QPoint
from PyQt4.QtGui import QInputDialog,QMenu

from Vispa.Plugins.EventBrowser.EventBrowserTabController import EventBrowserTabController

class EdmBrowserTabController(EventBrowserTabController):
    
    def __init__(self, plugin):
        EventBrowserTabController.__init__(self, plugin)
        self._treeDepth=1
        
    #@staticmethod
    def staticSupportedFileTypes():
        """ Returns supported file type: root.
        """
        return [('root', 'EDM root file')]
    staticSupportedFileTypes = staticmethod(staticSupportedFileTypes)
    
    def updateViewMenu(self):
        EventBrowserTabController.updateViewMenu(self)
        self.plugin().expandToDepthAction().setVisible(True)
        self.plugin().boxContentAction().setVisible(False)
        self.disconnect(self.tab().centerView(), SIGNAL("doubleClicked"), self.onDoubleClicked)
        self.connect(self.tab().centerView(), SIGNAL("doubleClicked"), self.onDoubleClicked)

    def onDoubleClicked(self,object):
        logging.debug(__name__ + ": onDoubleClicked()")
        if not self.tab().centerView().isUpdated(object):
            self.dataAccessor().read(object)
            self.updateCenterView()

    def onTreeViewSelected(self,select):
        EventBrowserTabController.onTreeViewSelected(self,self.dataAccessor().read(select,self._treeDepth))

    def onSelected(self,select):
        EventBrowserTabController.onSelected(self,self.dataAccessor().read(select))

    def updateCenterView(self,propertyView=True):
        if self.tab().treeView().selection():
            self.dataAccessor().read(self.tab().treeView().selection(),self._treeDepth)
        EventBrowserTabController.updateCenterView(self,propertyView)

    def expandToDepthDialog(self):
        """ Show dialog and expand center view to depth.
        """
        if hasattr(QInputDialog, "getInteger"):
            # Qt 4.3
            (depth, ok) = QInputDialog.getInteger(self.tab(), "Expand to depth...", "Input depth:", self._treeDepth, 0)
        else:
            # Qt 4.5
            (depth, ok) = QInputDialog.getInt(self.tab(), "Expand to depth...", "Input depth:", self._treeDepth, 0)
        if ok:
            self._treeDepth=depth
            if self.tab().treeView().selection():
                self.onTreeViewSelected(self.tab().treeView().selection())

    def centerViewMenuButtonClicked(self, point=None):
        popup=QMenu(self.tab().centerViewMenuButton())
        popup.addAction(self.plugin()._expandToDepthAction)
        popup.addAction(self.plugin()._saveImageAction)
        popup.addAction(self.plugin()._zoomAction)
        popup.addAction(self.plugin()._filterAction)
        popup.addSeparator()
        for action in self.plugin().viewMenu().actions():
            if action.data().toString()!="":
                popup.addAction(action)
        if not isinstance(point,QPoint):
            point=self.tab().centerViewMenuButton().mapToGlobal(QPoint(self.tab().centerViewMenuButton().width(),0))
        popup.exec_(point)

    def toggleFilterBranches(self):
        if self.dataAccessor().setFilterBranches(not self.dataAccessor().filterBranches()):
            self.updateContent()
