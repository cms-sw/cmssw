import logging

from PyQt4.QtCore import SIGNAL,QPoint
from PyQt4.QtGui import QWidget,QMenu

from Vispa.Main.TabController import TabController
from Vispa.Gui.FindDialog import FindDialog
from Vispa.Share.FindAlgorithm import FindAlgorithm
from Vispa.Gui.BoxContentDialog import BoxContentDialog
from Vispa.Views.AbstractView import AbstractView
from Vispa.Plugins.Browser.BrowserTab import BrowserTab
from Vispa.Gui.Zoomable import Zoomable

class BrowserTabController(TabController):
    """ Controls a tab with a TreeView, an AbstractView and a PropertiesView.
    
    The tab is filled using a DataAccessor. The controller supports find
    functionality as well as a dialog for choosing the box content in the WidgetView.
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        TabController.__init__(self, plugin)

        self._dataAccessor = None

        self.setFindEnabled()
        self._findAlgorithm = None
        self._findDialog = None
        self._filterAlgoritm = None
        self._filterDialog = None
        self._filterObjects = None
        self._boxContentDialog = BoxContentDialog(self.plugin().application().mainWindow())
        self.connect(self._boxContentDialog, SIGNAL("scriptChanged"), self.scriptChanged)

    def centerView(self):
        if self.tab():
            return self.tab().centerView()
        return None
    
    def currentCenterViewClassId(self):
        if self.tab() and self.tab().centerView():
            return self.plugin().viewClassId(self.tab().centerView().__class__)
        return None

    def enableCenterViewSelectionMenu(self, enable=True, exceptionViewClassId=None):
        disabledCenterViewIds = []
        for viewClass in self.plugin().availableCenterViews():
            viewClassId=self.plugin().viewClassId(viewClass)
            if enable==False and viewClassId!=exceptionViewClassId:
                disabledCenterViewIds+=[viewClassId]
            if enable==True and viewClassId==exceptionViewClassId:
                disabledCenterViewIds+=[viewClassId]
        self.plugin().setDisabledCenterViewIds(disabledCenterViewIds)

    def activated(self):
        """ Shows view menu when user selects tab.
        """
        self.updateViewMenu()
        self.tab().mainWindow().application().showPluginMenu(self.plugin().viewMenu())

    def switchCenterView(self, requestedViewClassId):
        if self.currentCenterViewClassId() == requestedViewClassId:
            self.updateViewMenu()
            return True
        requestedViewClass = None
        for viewClass in self.plugin().availableCenterViews():
            if requestedViewClassId == self.plugin().viewClassId(viewClass):
                requestedViewClass = viewClass
        if not requestedViewClass and len(self.plugin().availableCenterViews())>0:
            logging.warning(self.__class__.__name__ +": switchCenterView() - Unknown view class id "+ requestedViewClassId +".")
            requestedViewClass = self.plugin().availableCenterViews()[0]
        elif not requestedViewClass:
            logging.error(self.__class__.__name__ +": switchCenterView() - Unknown view class id "+ requestedViewClassId +". Aborting...")
            return False
        self.tab().setCenterView(requestedViewClass())
        self.updateViewMenu()

        #reconnect data accessors and stuff
        if hasattr(self.tab().centerView(), "setEditable"):
            self.tab().centerView().setEditable(self.isEditable())
        self.tab().centerView().setDataAccessor(self._dataAccessor)
        self.tab().centerView().setFilter(self.filter)
        selection = self.tab().treeView().selection()
        if selection:
            self.tab().centerView().setDataObjects([selection])
        self.updateCenterView()
        self.connect(self.tab().centerView(), SIGNAL("selected"), self.onSelected)
        self.connect(self.tab().centerView(), SIGNAL("modified"), self.setModified)
        self.connect(self.tab().centerView(), SIGNAL("mouseRightPressed"), self.centerViewMenuButtonClicked)
        self.saveIni()
        return True
                        
    def showBoxContentDialog(self):
        self._boxContentDialog.onScreen()
        
    def setEditable(self, edit):
        """ Makes sure an existing property view's read-only mode is set accordingly.
        """
        TabController.setEditable(self, edit)
        if self.tab() and self.tab().propertyView():
            self.tab().propertyView().setReadOnly(not edit)
            
    def setTab(self, tab):
        """ Sets tab and connects signals to tab. 
        """
        if not (isinstance(tab.treeView(), AbstractView) and isinstance(tab.treeView(), QWidget)):
            raise TypeError(__name__ + " requires a center view of type AbstractView and QWidget.")
        if not (isinstance(tab.centerView(), AbstractView) and isinstance(tab.centerView(), QWidget)):
            raise TypeError(__name__ + " requires a center view of type AbstractView and QWidget.")
        if not isinstance(tab, BrowserTab):
            raise TypeError(__name__ + " requires a tab of type BrowserTab.")
        TabController.setTab(self, tab)
        self.connect(self.tab().treeView(), SIGNAL("selected"), self.onTreeViewSelected)
        self.connect(self.tab().treeView(), SIGNAL("mouseRightPressed"), self.treeViewMenuButtonClicked)
        self.connect(self.tab().centerView(), SIGNAL("selected"), self.onSelected)
        self.connect(self.tab().centerView(), SIGNAL("mouseRightPressed"), self.centerViewMenuButtonClicked)
        self.connect(self.tab().treeViewHeader(), SIGNAL("mouseRightPressed"), self.treeViewMenuButtonClicked)
        self.connect(self.tab().centerViewHeader(), SIGNAL("mouseRightPressed"), self.centerViewMenuButtonClicked)
        
        if self._dataAccessor:
            # make sure sub-components of tab also know controller
            self.setDataAccessor(self._dataAccessor)
        
        self.loadIni()
        
    def setDataAccessor(self, accessor):
        """ Set the DataAccessor and show data in the TreeView.
        """
        logging.debug(__name__ + ": setDataAccessor")
        self._dataAccessor = accessor
        if self.tab():
            self.tab().treeView().setDataAccessor(self._dataAccessor)
            self.tab().centerView().setDataAccessor(self._dataAccessor)
            self.tab().propertyView().setDataAccessor(self._dataAccessor)
            self.tab().treeView().setFilter(self.filter)
            self.tab().centerView().setFilter(self.filter)

    def dataAccessor(self):
        return self._dataAccessor

    def setZoom(self, zoom):
        """  Sets zoom of tab's scroll area.
        
        Needed for zoom tool bar. See TabController setZoom().
        """
        if hasattr(self.tab(),"scrollArea"):
            self.tab().scrollArea().setZoom(zoom)
    
    def zoom(self):
        """ Returns zoom of tab's scoll area.
        
        Needed for zoom tool bar. See TabController zoom().
        """
        if hasattr(self.tab(),"scrollArea"):
            return self.tab().scrollArea().zoom()
        else:
            return 100.0

    def updateCenterView(self,propertyView=True):
        """ Fill the center view from an item in the TreeView and update it """
        logging.debug(__name__ + ": updateCenterView")
        statusMessage = self.plugin().application().startWorking("Updating center view")
        if self.tab().centerView().updateContent():
            self.tab().centerView().restoreSelection()
            select = self.tab().centerView().selection()
            if select != None:
                if self.tab().propertyView().dataObject() != select and propertyView:
                    self.tab().propertyView().setDataObject(select)
                    self.tab().propertyView().updateContent()
        self.plugin().application().stopWorking(statusMessage)

    def onTreeViewSelected(self, select):
        """ When object is selected in the TreeView update center view and PropertyView.
        """
        logging.debug(__name__ + ": onTreeViewSelected")
        self.onSelected(select)
        self.tab().centerView().setDataObjects([self.tab().treeView().selection()])
        self.updateCenterView()

    def onSelected(self, select):
        """ When object is selected in the center view update PropertyView.
        """
        logging.debug(__name__ + ": onSelected")
        if self.tab().propertyView().dataObject() != select:
            statusMessage = self.plugin().application().startWorking("Updating property view")
            self.tab().propertyView().setDataObject(select)
            self.tab().propertyView().updateContent()
            self.plugin().application().stopWorking(statusMessage)

    def updateContent(self, filtered=False, propertyView=True):
        """ Updates all three views and restores the selection, e.g. after moving to next event.
        """
        logging.debug(__name__ + ": updateContent")
        # run filter if used
        if self._filterDialog and not filtered:
            self._filterAlgoritm.setDataObjects(self._dataAccessor.topLevelObjects())
            self._filterDialog.filter()
            return
        statusMessage = self.plugin().application().startWorking("Updating all views")
        if self._findAlgorithm:
            self._findAlgorithm.setDataObjects(self._dataAccessor.topLevelObjects())
            self._findDialog.edited()
        self.tab().treeView().setDataObjects(self._dataAccessor.topLevelObjects())
        if self.updateTreeView():
            if propertyView:
                self.tab().propertyView().setDataObject(self.tab().treeView().selection())
            if not propertyView or self.tab().propertyView().updateContent():
                selection = self.tab().treeView().selection()
                if selection:
                    self.tab().centerView().setDataObjects([selection])
                self.updateCenterView(propertyView)
        self.plugin().application().stopWorking(statusMessage)

    def updateTreeView(self):
        if self.tab().treeView().updateContent():
            self.tab().treeView().restoreSelection()
            return True
        else:
            return False
    
    def find(self):
        """ Open find dialog and find items.
        """
        logging.debug(__name__ + ": find")
        if not self._findAlgorithm:
            self._findAlgorithm = FindAlgorithm()
            self._findAlgorithm.setDataAccessor(self._dataAccessor)
            self._findAlgorithm.setFilter(self.filter)
        if not self._findDialog:
            self._findDialog = FindDialog(self.tab())
            self._findDialog.setFindAlgorithm(self._findAlgorithm)
            self.connect(self._findDialog, SIGNAL("found"), self.select)
        self._findAlgorithm.setDataObjects(self._dataAccessor.topLevelObjects())
        self._findDialog.onScreen()
        
    def filterDialog(self):
        """ Open filter dialog and filter items.
        """
        logging.debug(__name__ + ": filterDialog")
        if not self._filterAlgoritm:
            self._filterAlgoritm = FindAlgorithm()
            self._filterAlgoritm.setDataAccessor(self._dataAccessor)
        if not self._filterDialog:
            self._filterDialog = FindDialog(self.tab())
            self._filterDialog.setFindAlgorithm(self._filterAlgoritm)
            self.connect(self._filterDialog, SIGNAL("filtered"), self.filtered)
        self._filterAlgoritm.setDataObjects(self._dataAccessor.topLevelObjects())
        self._filterDialog.onScreen(True, False)
    
    def filtered(self, filterObjects):
        self._filterObjects = filterObjects
        self.updateContent(True)

    def select(self, object):
        """ Select an object in all views.
        """
        logging.debug(__name__ + ": select : " + str(object))
        self.tab().treeView().select(object)
        self.tab().centerView().select(object)
        if self.tab().propertyView().dataObject() != object:
            statusMessage = self.plugin().application().startWorking("Updating property view")
            self.tab().propertyView().setDataObject(object)
            self.tab().propertyView().updateContent()
            self.plugin().application().stopWorking(statusMessage)

    def scriptChanged(self, script):
        """ Update box content of center view when script is changed.
        """
        if hasattr(self.tab().centerView(), "setBoxContentScript"):
            self.tab().centerView().setBoxContentScript(script)
            self.updateCenterView()

    def close(self):
        self.cancel()
        return TabController.close(self)
    
    def boxContentDialog(self):
        return self._boxContentDialog

    def saveImage(self, filename=None):
        """ Save screenshot of the center view to file.
        """
        self.tab().centerView().exportImage(filename)

    def filter(self, objects):
        """ Filter all final state objects using the output of the filterDialog.
        """
        #logging.debug(__name__ + ": filter")
        if self._filterObjects != None:
            return [o for o in objects if o in self._filterObjects or len(self.filter(self._dataAccessor.children(o)))>0]
        else:
            return objects

    def cancel(self):
        """ Cancel all operations in tab.
        """
        logging.debug(__name__ + ": cancel")
        self.tab().treeView().cancel()
        self.tab().centerView().cancel()
        self.tab().propertyView().cancel()

    def isBusy(self):
        return self.tab().treeView().isBusy() or\
               self.tab().centerView().isBusy() or\
               self.tab().propertyView().isBusy()

    def saveIni(self):
        """ write options to ini """
        logging.debug(__name__ + ": saveIni")
        if not self.plugin():
            logging.waring(self.__class__.__name__ +": saveIni() - No plugin set. Aborting...")
            return
        ini = self.plugin().application().ini()
        if not ini.has_section("view"):
            ini.add_section("view")
        if self.currentCenterViewClassId():
            ini.set("view", "CurrentView", self.currentCenterViewClassId())
        if hasattr(self.centerView(), "boxContentScript"):
            ini.set("view", "box content script", self.centerView().boxContentScript())
        self.plugin().application().writeIni()

    def loadIni(self):
        """ read options from ini """
        logging.debug(__name__ + ": loadIni")
        ini = self.plugin().application().ini()
        if ini.has_option("view", "CurrentView"):
            proposed_view = ini.get("view", "CurrentView")
            self.switchCenterView(proposed_view)
        elif self.plugin().defaultCenterViewId():
            self.switchCenterView(self.plugin().defaultCenterViewId())
        elif len(self.plugin().availableCenterViews()) > 0:
            self.switchCenterView(self.plugin().viewClassId(self.plugin().availableCenterViews()[0]))
        if ini.has_option("view", "box content script"):
            self._boxContentDialog.setScript(str(ini.get("view", "box content script")))
            if hasattr(self.centerView(), "setBoxContentScript"):
                self.centerView().setBoxContentScript(str(ini.get("view", "box content script")))

    def updateViewMenu(self):
        """ Enable/disable menu entries, when center view changes.
        """
        self.plugin().boxContentAction().setVisible(hasattr(self.tab().centerView(),"setBoxContentScript"))
        self.plugin().saveImageAction().setVisible(hasattr(self.tab().centerView(),"exportImage"))
        self.plugin().zoomAction().setVisible(hasattr(self.tab().centerView(),"setZoom"))
        self.plugin().expandAllAction().setVisible(hasattr(self.tab().treeView(),"expandAll"))
        self.plugin().expandToDepthAction().setVisible(hasattr(self.tab().treeView(),"expandToDepth"))
        self.plugin().collapseAllAction().setVisible(hasattr(self.tab().treeView(),"collapseAll"))
        for action in self.plugin().viewMenu().actions():
            if action.data().toString()!="":
                action.setEnabled(not action.data().toString() in self.plugin().disabledCenterViewIds())
                currentAction=action.data().toString()==self.currentCenterViewClassId()
                action.setChecked(currentAction)
                if currentAction:
                    self.tab().setCenterViewHeader(action.text().replace("&",""))
        if self.tab().mainWindow():
            if isinstance(self.tab().centerView(), Zoomable):
                self.tab().mainWindow().application().showZoomToolBar()
            else:
                self.tab().mainWindow().application().hideZoomToolBar()

    def centerViewMenuButtonClicked(self, point=None):
        popup=QMenu(self.tab().centerViewMenuButton())
        popup.addAction(self.plugin()._boxContentAction)
        popup.addAction(self.plugin()._saveImageAction)
        popup.addAction(self.plugin()._zoomAction)
        popup.addSeparator()
        for action in self.plugin().viewMenu().actions():
            if action.data().toString()!="":
                popup.addAction(action)
        if not isinstance(point,QPoint):
            point=self.tab().centerViewMenuButton().mapToGlobal(QPoint(self.tab().centerViewMenuButton().width(),0))
        popup.exec_(point)

    def treeViewMenuButtonClicked(self, point=None):
        popup=QMenu(self.tab().treeViewMenuButton())
        popup.addAction(self.plugin()._expandAllAction)
        popup.addAction(self.plugin()._expandToDepthAction)
        popup.addAction(self.plugin()._collapseAllAction)
        popup.addAction(self.plugin()._filterAction)
        popup.addSeparator()
        if not isinstance(point,QPoint):
            point=self.tab().treeViewMenuButton().mapToGlobal(QPoint(self.tab().treeViewMenuButton().width(),0))
        popup.exec_(point)
