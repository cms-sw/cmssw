import logging

from Vispa.Main.TabController import *
from Vispa.Main.FindDialog import *
from Vispa.Main.FindAlgorithm import *
from Vispa.Main.BoxContentDialog import *
from Vispa.Main.EventFileNavigator import *
from Vispa.Main.WidgetView import *


class BrowserController(TabController):
    """ Controls a tab with a TreeView, a WidgetView and a PropertiesView.
    
    The tab is filled using a DataAccessor. The controller supports find
    functionality as well as a dialog for choosing the box content in the WidgetView.
    """
    def __init__(self, plugin, enableCenterView=True):
        logging.debug(__name__ + ": __init__")
        TabController.__init__(self, plugin)

        self._dataAccessor = None

        #This dict stores the available center views
        #The idea is that it is easy to add center views, and there is a
        #common interface to switch between them
        self._availableCenterViews = {}
        self._centerViewActions = {}
        self._currentCenterView = None
        self._enableCenterView = enableCenterView
        
        self.setFindEnabled()
        self._findAlgoritm = None
        self._findDialog = None
        self._filterAlgoritm = None
        self._filterDialog = None
        self._filterObjects = None
        self._boxContentDialog = BoxContentDialog(self.plugin().application().mainWindow())
        self.connect(self._boxContentDialog, SIGNAL("scriptChanged"), self.scriptChanged)

        self._fillMenu()
  
    def currentCenterView(self):
        return self._currentCenterView

    def _fillMenu(self):
        """ Fill specific menu.
        """
        self._viewMenu = self.plugin().application().createPluginMenu('&View')
        self._expandTreeAction = self.plugin().application().createAction('&Expand tree', self._expandTree, "Ctrl+E")
        self._viewMenu.addAction(self._expandTreeAction)
        self._collapseTreeAction = self.plugin().application().createAction('&Collapse tree', self._collapseTree, "Ctrl+L")
        self._viewMenu.addAction(self._collapseTreeAction)
        self._viewMenu.addSeparator()
        self._filterAction = self.plugin().application().createAction('&Apply filter...', self.filterDialog, "Ctrl+P")
        self._viewMenu.addAction(self._filterAction)
        if self._enableCenterView:
            self._boxContentAction = self.plugin().application().createAction('&Set box content...', self._showBoxContentDialog, "Ctrl+B")
            self._viewMenu.addAction(self._boxContentAction)
            self._saveImageAction = self.plugin().application().createAction('&Save image...', self.saveImage, "Ctrl+I")
            self._viewMenu.addAction(self._saveImageAction)
            self._zoomAction = self.plugin().application().createAction('&Zoom...', self.zoomDialog, "Ctrl+Z")
            self._viewMenu.addAction(self._zoomAction)
            self._centerViewSelectMenu = self._viewMenu.addMenu('&Center View')
        
    def addCenterView(self, name, function, checked=False, shortcut=None,enabled= True):
        '''add a View for the Center View
         selection: name is the menu entry, function the function to be
         added as action if the menu is selecteds'''
      
        logging.debug(__name__ + ": addCenterView")
        if self._availableCenterViews.has_key(name) and self._availableCenterViews[name] != None:
            logging.warning(__name__ + ": " + name + " Already have a View of this name! Overwritiong old View!")
        self._availableCenterViews[name] = function 
        self._centerViewActions[name] = self.plugin().application().createAction(name, self._switchCenterView, shortcut)
        self._centerViewActions[name].setCheckable(True)
        self._centerViewActions[name].setEnabled(enabled)
        if checked:
            self._centerViewActions[name].setChecked(True)
            self._currentCenterView = name

    def fillCenterViewSelectMenu(self):
        logging.debug(__name__ + ": fillCenterViewSelectMenu")
        for action in self._centerViewActions.values():
            self._centerViewSelectMenu.addAction(action)
    
    def _switchCenterView(self):
        '''switches to centerview name - name is the key of
        availeableCenterViews
        '''
        #check if view is checked again
        if self._centerViewActions[self._currentCenterView].isChecked():
            self._centerViewActions[self._currentCenterView].setChecked(False)
        else:
            self._centerViewActions[self._currentCenterView].setChecked(True)

        for view in self._availableCenterViews.keys():
            if self._centerViewActions[view].isChecked():
                self._currentCenterView = view
                break
      
        #run function to update center view
        self._availableCenterViews[self._currentCenterView]()
        self.connect(self.tab().centerView(), SIGNAL("selected"), self.onSelected)
        #reconnect data accessors and stuff
        self.tab().centerView().setDataAccessor(self._dataAccessor)
        self.tab().centerView().setFilter(self.filter)
        self.updateCenterView()

    def viewMenu(self):
        return self._viewMenu

    def _showBoxContentDialog(self):
        self._boxContentDialog.onScreen()
        
    def _expandTree(self):
        self.tab().treeView().expandAll()
        
    def _collapseTree(self):
        self.tab().treeView().collapseAll()
        
    def setTab(self, tab):
        """ Sets tab and connects signals to tab. 
        """
        if not isinstance(tab.centerView(), WidgetView):
            raise TypeError(__name__ + " requires a center view of type WidgetView.")
        TabController.setTab(self, tab)
        self.connect(self.tab().treeView(), SIGNAL("selected"), self.onTreeViewSelected)
        self.connect(self.tab().centerView(), SIGNAL("selected"), self.onSelected)
        
    def setDataAccessor(self, accessor):
        """ Set the DataAccessor and show data in the TreeView.
        """
        logging.debug(__name__ + ": setDataAccessor")
        self._dataAccessor = accessor
        self.tab().treeView().setDataAccessor(self._dataAccessor)
        self.tab().centerView().setDataAccessor(self._dataAccessor)
        self.tab().propertyView().setDataAccessor(self._dataAccessor)
        self.tab().treeView().setFilter(self.filter)
        self.tab().centerView().setFilter(self.filter)

    def dataAccessor(self):
        return self._dataAccessor

    def selected(self):
        """ Shows plugin menus when user selects tab.
        """
        logging.debug(__name__ + ": selected()")
        self.plugin().application().showPluginMenu(self._viewMenu)
        
    def setZoom(self, zoom):
        """  Sets zoom of tab's scroll area.
        
        Needed for zoom tool bar. See TabController setZoom().
        """
        self.tab().scrollArea().setZoom(zoom)
    
    def zoom(self):
        """ Returns zoom of tab's scoll area.
        
        Needed for zoom tool bar. See TabController zoom().
        """
        return self.tab().scrollArea().zoom()

    def updateCenterView(self):
        """ Fill the center view from an item in the TreeView and update it """
        logging.debug(__name__ + ": updateCenterView")
        statusMessage = self.plugin().application().startWorking("Updating center view")
        self.tab().centerView().setDataObjects([self.tab().treeView().selection()])
        result = self.tab().centerView().updateContent()
        if result:
            self.tab().centerView().restoreSelection()
            select = self.tab().centerView().selection()
            if select != None:
                if self.tab().propertyView().dataObject() != select:
                    self.tab().propertyView().setDataObject(select)
                    self.tab().propertyView().updateContent()
        self.plugin().application().stopWorking(statusMessage)
        return result

    def onTreeViewSelected(self, select):
        """ When object is selected in the TreeView update center view and PropertyView.
        """
        logging.debug(__name__ + ": onTreeViewSelected")
        self.onSelected(select)
        self.tab().centerView().select(None)
        self.updateCenterView()

    def onSelected(self, select):
        """ When object is selected in the center view update PropertyView.
        """
        logging.debug(__name__ + ": onSelected")
        if self.tab().propertyView().dataObject() != select:
            self.tab().propertyView().setDataObject(select)
            self.tab().propertyView().updateContent()

    def updateContent(self, filtered=False):
        """ Updates all three views and restores the selection, e.g. after moving to next event.
        """
        logging.debug(__name__ + ": updateContent")
        # run filter if used
        if self._filterDialog and not filtered:
            self._filterAlgoritm.setDataObjects(self._dataAccessor.topLevelObjects())
            self._filterDialog.filter()
            return
        self.tab().treeView().setDataObjects(self._dataAccessor.topLevelObjects())
        self.tab().treeView().updateContent()
        self.tab().treeView().restoreSelection()
        self.tab().propertyView().setDataObject(self.tab().treeView().selection())
        self.tab().propertyView().updateContent()
        self.updateCenterView()
    
    def find(self):
        """ Open find dialog and find items.
        """
        logging.debug(__name__ + ": find")
        if not self._findAlgoritm:
            self._findAlgoritm = FindAlgorithm()
            self._findAlgoritm.setDataAccessor(self._dataAccessor)
            self._findAlgoritm.setFilter(self.filter)
        if not self._findDialog:
            self._findDialog = FindDialog(self.tab())
            self._findDialog.setFindAlgorithm(self._findAlgoritm)
            self.connect(self._findDialog, SIGNAL("found"), self.select)
        self._findAlgoritm.setDataObjects(self._dataAccessor.topLevelObjects())
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
            self.tab().propertyView().setDataObject(object)
            self.tab().propertyView().updateContent()

    def scriptChanged(self, script):
        """ Update box content of center view when script is changed.
        """
        if hasattr(self.tab().centerView(), "setBoxContentScript"):
            self.tab().centerView().setBoxContentScript(script)
            self.updateCenterView()

    def closeEvent(self, event):
        event.accept()

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
        if self.filterDialog and self._filterObjects != None:
            return [o for o in objects if o in self._filterObjects or len(self.filter(self._dataAccessor.children(o)))>0]
        else:
            return objects
