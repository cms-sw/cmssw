import logging
import os.path

from PyQt4.QtCore import SIGNAL,QPoint
from PyQt4.QtGui import QInputDialog,QMenu

from Vispa.Plugins.EventBrowser.EventBrowserTabController import EventBrowserTabController
from Vispa.Plugins.EdmBrowser.EventContentDialog import EventContentDialog

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
        self.disconnect(self.tab().centerView(), SIGNAL("toggleCollapsed"), self.toggleCollapsed)
        self.connect(self.tab().centerView(), SIGNAL("toggleCollapsed"), self.toggleCollapsed)

    def toggleCollapsed(self,object):
        logging.debug(__name__ + ": toggleCollapsed()")
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
        self.dataAccessor().setFilterBranches(not self.dataAccessor().filterBranches())
        self.updateContent()
        self.saveIni()

    def toggleUnderscoreProperties(self):
        self.dataAccessor().setUnderscoreProperties(not self.dataAccessor().underscoreProperties())
        self.tab().propertyView().setDataObject(None)
        self.updateCenterView()
        self.saveIni()

    def loadIni(self):
        """ read options from ini """
        EventBrowserTabController.loadIni(self)
        ini = self.plugin().application().ini()
        if ini.has_option("edm", "filter branches"):
            self.dataAccessor().setFilterBranches(ini.get("edm", "filter branches")=="True")
        if ini.has_option("edm", "underscore properties"):
            self.dataAccessor().setUnderscoreProperties(ini.get("edm", "underscore properties")=="True")
        self.plugin().hideUnderscorePropertiesAction().setChecked(not self.dataAccessor().underscoreProperties())

    def saveIni(self):
        """ write options to ini """
        EventBrowserTabController.saveIni(self)
        ini = self.plugin().application().ini()
        if not ini.has_section("edm"):
            ini.add_section("edm")
        ini.set("edm", "filter branches", self.dataAccessor().filterBranches())
        ini.set("edm", "underscore properties", self.dataAccessor().underscoreProperties())
        self.plugin().application().writeIni()

    def eventContent(self):
        """ Open event content dialog """
        logging.debug(__name__ + ": eventContent")
        dialog=EventContentDialog(self.tab(),"This dialog let's you compare the contents of your edm root file with another dataformat / edm root file. You can compare either to a dataformat definition from a txt file (e.g. RECO_3_3_0) or any edm root file by selecting an input file.")
        branches=[branch[0].split("_") for branch in self.dataAccessor().filteredBranches()]
        name = os.path.splitext(os.path.basename(self._filename))[0]
        dialog.setEventContent(name,branches)
        dialog.exec_()
