import logging

from Vispa.Main.TabController import *
from Vispa.Main.FindDialog import *
from Vispa.Main.FindAlgorithm import *
from Vispa.Main.BoxContentDialog import *
from Vispa.Main.EventFileNavigator import *
from Vispa.Main.Workspace import *

class TripleTabController(TabController):
    """ Controls a tab with a TreeView, a Workspace and a PropertiesView.
    
    The tab is filled using a DataAccessor. The controller supports find
    functionality as well as a dialog for choosing the box content in the workspace.
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        TabController.__init__(self, plugin)

        self._dataAccessor = None
        self._treeViewSelection = None
        self._centerViewSelection = None
        
        #This dict stores the available center views
        #The idea is that it is easy to add center views, and there is a
        #common interface to switch between them
        self._availableCenterViews = {}
        self._centerViewActions = {}

        self._currentCenterView = None
        
        self._restoreSelectionFlag = False
        
        self.setFindEnabled()
        self._findDialog = None

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
        self._boxContentAction = self.plugin().application().createAction('&Set box content...', self._showBoxContentDialog, "Ctrl+B")
        self._viewMenu.addAction(self._boxContentAction)
        self._saveImageAction = self.plugin().application().createAction('&Save image...', self.saveImage, "Ctrl+I")
        self._viewMenu.addAction(self._saveImageAction)
        self._zoomAction = self.plugin().application().createAction('&Zoom...', self.zoomDialog, "Ctrl+Z")
        self._viewMenu.addAction(self._zoomAction)
    
        self._centerViewSelectMenu = self._viewMenu.addMenu('&Center View')
        

    def addCenterView(self, name, function, checked=False, shortcut=None):
      '''add a View for the Center View
      selection: name is the menu entry, function the function to be
      added as action if the menu is selecteds'''
      
      logging.debug(__name__ + ": addCenterView")
      if self._availableCenterViews.has_key(name) and self._availableCenterViews[name] != None:
        logging.warning(__name__ + ": " + name + " Already have a View of this name! Overwritiong old View!")
      self._availableCenterViews[name] = function 
      self._centerViewActions[name] = self.plugin().application().createAction(name, self._switchCenterView, shortcut)
      self._centerViewActions[name].setCheckable(True)
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
     
      #self._boxContentDialog = BoxContentDialog(self.tab())
      self.connect(self.tab().centerView(), SIGNAL("widgetSelected"), self.onWidgetSelected)
        
      #reconnect data accessors and stuff
      self.tab().centerView().setDataAccessor(self._dataAccessor)
      self.updateAndRestoreSelection() 
      
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
        if not isinstance(tab.centerView(), Workspace):
            raise TypeError(__name__ + " requires a center view of type Workspace.")
        TabController.setTab(self, tab)
        self._boxContentDialog = BoxContentDialog(self.tab())
        self.connect(self.tab().treeView(), SIGNAL("itemSelected"), self.onItemSelected)
        self.connect(self.tab().centerView(), SIGNAL("widgetSelected"), self.onWidgetSelected)
        self.connect(self._boxContentDialog, SIGNAL("scriptChanged"), self.scriptChanged)
        
    def setDataAccessor(self, accessor):
        """ Set the DataAccessor and show data in the TreeView.
        """
        logging.debug(__name__ + ": setDataAccessor")
        self._dataAccessor = accessor
        self.tab().treeView().setDataAccessor(self._dataAccessor)
        self.tab().centerView().setDataAccessor(self._dataAccessor)
        self.tab().propertyView().setDataAccessor(self._dataAccessor)

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

    def updateCenterView(self, item):
        """ Fill the center view from an item in the TreeView and update it """
        statusMessage = self.plugin().application().startWorking("Updating center view")
        if item != None:
            self.tab().centerView().setDataObjects([item.object])
        else:
            self.tab().centerView().setDataObjects([])
        self.tab().centerView().updateContent()
        self.plugin().application().stopWorking(statusMessage)

    def onItemSelected(self, item):
        """ When item is selected in the TreeView update center view and PropertyView.
        """
        if self._restoreSelectionFlag:
            return False
        logging.debug(__name__ + ": onItemSelected")
        self._treeViewSelection = item.itemId
        if self.tab().propertyView().dataObject() != item.object:
            self.tab().propertyView().setDataObject(item.object)
            self.tab().propertyView().updateContent()
        self.updateCenterView(item)
        # Select first item in DecayTreeView
        self.tab().centerView().select(self.tab().centerView().widgetById("0"))

    def onWidgetSelected(self, widget):
        """ When widget is selected in the center view update PropertyView.
        """
        if self._restoreSelectionFlag:
            return False
        logging.debug(__name__ + ": onWidgetSelected")
        self._centerViewSelection = widget.widgetId
        if self.tab().propertyView().dataObject() != widget.object:
            self.tab().propertyView().setDataObject(widget.object)
            self.tab().propertyView().updateContent()

    def updateAndRestoreSelection(self):
        """ Updates all three views and restores the selection, e.g. after moving to next event.
        """
        self._restoreSelectionFlag = True
        self.tab().treeView().setDataObjects(self._dataAccessor.topLevelObjects())
        self.tab().treeView().updateContent()
        if self._treeViewSelection != None:
            self.tab().propertyView().setDataObject(None)
            selectedItem = self.tab().treeView().itemById(self._treeViewSelection)
            self.tab().treeView().select(selectedItem)
            self.updateCenterView(selectedItem)
            if selectedItem != None:
                self.tab().propertyView().setDataObject(selectedItem.object)
            if self._centerViewSelection != None:
                selectedWidget = self.tab().centerView().widgetById(self._centerViewSelection)
                self.tab().centerView().select(selectedWidget)
                if selectedWidget != None:
                    self.tab().propertyView().setDataObject(selectedWidget.object)
            self.tab().propertyView().updateContent()
            self._restoreSelectionFlag = False
        else:
            self._restoreSelectionFlag = False
            self.tab().treeView().select(self.tab().treeView().itemById("0"))
        
    def find(self):
        """ Open find dialog and find items.
        """
        logging.debug(__name__ + ": find")
        if not self._findDialog:
            self._findAlgoritm = FindAlgorithm()
            self._findAlgoritm.setDataAccessor(self._dataAccessor)
            self._findAlgoritm.setDataObjects(self._dataAccessor.topLevelObjects())
            self._findDialog = FindDialog(self.tab())
            self._findDialog.setFindAlgorithm(self._findAlgoritm)
            self.connect(self._findDialog, SIGNAL("found"), self.select)
        self._findDialog.onScreen()
        
    def select(self, object):
        """ Select an object in all views.
        """
        logging.debug(__name__ + ": select : " + str(object))
        self._restoreSelectionFlag = True
        selectedItem = self.tab().treeView().itemByObject(object)
        self.tab().treeView().select(selectedItem)
        selectedWidget = self.tab().centerView().widgetByObject(object)
        self.tab().centerView().select(selectedWidget)
        self.tab().propertyView().setDataObject(object)
        self.tab().propertyView().updateContent()
        self._restoreSelectionFlag = False

    def scriptChanged(self, script):
        """ Update box content of center view when script is changed.
        """
        self._restoreSelectionFlag = True
        self.tab().centerView().setBoxContentScript(script)
        self.tab().centerView().updateContent()
        if self._centerViewSelection != None:
            selectedWidget = self.tab().centerView().widgetById(self._centerViewSelection)
            self.tab().centerView().select(selectedWidget)
        self._restoreSelectionFlag = False

    def closeEvent(self, event):
        event.accept()

    def boxContentDialog(self):
        return self._boxContentDialog

    def saveImage(self, filename=None):
        """ Save screenshot of the center view to file.
        """
        self.tab().centerView().exportImage(filename)

    def zoomDialog(self):
        if hasattr(QInputDialog, "getInteger"):
            # Qt 4.3
            (zoom, ok) = QInputDialog.getInteger(self.tab(), "Zoom...", "Input zoom factor in percent:", self.zoom(), 0)
        else:
            # Qt 4.5
            (zoom, ok) = QInputDialog.getInt(self.tab(), "Zoom...", "Input zoom factor in percent:", self.zoom(), 0)
        if ok:
            self.setZoom(zoom)
