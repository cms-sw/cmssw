import os.path
import logging

from PyQt4.QtCore import QVariant

from Vispa.Main.VispaPlugin import VispaPlugin
from Vispa.Views.AbstractView import AbstractView
from Vispa.Main.Exceptions import NoCurrentTabControllerException

class BrowserPlugin(VispaPlugin):
    """ The BrowserPlugin supplies the view menu and supports center views.
    """
    
    def __init__(self, application=None):
        logging.debug(__name__ + ": __init__")
        VispaPlugin.__init__(self, application)
        self._defaultCenterViewId = None
        self._disabledCenterViewIds = []
        self._availableCenterViews = []
        self._startUp=True
    
    def startUp(self):
        self._startUp=False
        self._fillMenu()

    def defaultCenterViewId(self):
        return self._defaultCenterViewId

    def disabledCenterViewIds(self):
        return self._disabledCenterViewIds

    def setDisabledCenterViewIds(self,ids):
        self._disabledCenterViewIds=ids

    def availableCenterViews(self):
        return self._availableCenterViews

    def _fillMenu(self):
        """ Fill specific menu.
        """
        self._viewMenu = self.application().createPluginMenu('&View')
        self._expandAllAction = self.application().createAction('&Expand all', self._expandAll,"Ctrl+E")
        self._viewMenu.addAction(self._expandAllAction)
        self._expandToDepthAction = self.application().createAction('Expand to &depth...', self._expandToDepth, "Ctrl+Shift+E")
        self._viewMenu.addAction(self._expandToDepthAction)
        self._collapseAllAction = self.application().createAction('&Collapse all', self._collapseAll,"Ctrl+L")
        self._viewMenu.addAction(self._collapseAllAction)
        self._viewMenu.addSeparator()
        self._filterAction = self.application().createAction('&Apply filter...', self.filterDialog, "Ctrl+P")
        self._viewMenu.addAction(self._filterAction)
        self._boxContentAction = self.application().createAction('Show &object details...', self._showBoxContentDialog, "Ctrl+B")
        self._viewMenu.addAction(self._boxContentAction)
        self._saveImageAction = self.application().createAction('&Save image...', self.saveImage, "Ctrl+I")
        self._viewMenu.addAction(self._saveImageAction)
        self._zoomAction = self.application().createAction('&Zoom...', self.zoomDialog, "Ctrl+Shift+Z")
        self._viewMenu.addAction(self._zoomAction)
        self._viewMenu.addSeparator()
    
    def viewMenu(self):
        return self._viewMenu
    
    def _expandAll(self):
        """ Calls expandAll() function of tree view.
        """
        try:
            self.application().currentTabController().tab().treeView().expandAll()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": _expandAll() - No tab controller found.")

    def _expandToDepth(self):
        """ Calls expandToDepthDialog() function of current tab controller.
        """
        try:
            self.application().currentTabController().tab().treeView().expandToDepthDialog()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": _expandToDepth() - No tab controller found.")

    def _collapseAll(self):
        """ Calls collapseAll() function of tree view.
        """
        try:
            self.application().currentTabController().tab().treeView().collapseAll()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": _collapseAll() - No tab controller found.")
   
    def filterDialog(self):
        """ Calls filterDialog() function of current tab controller.
        """
        try:
            self.application().currentTabController().filterDialog()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": filterDialog() - No tab controller found.")
   
    def _showBoxContentDialog(self):
        """ Calls showBoxContentDialog() function of current tab controller.
        """
        try:
            self.application().currentTabController().showBoxContentDialog()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": showBoxContentDialog() - No tab controller found.")
   
    def zoomDialog(self):
        """ Calls zoomDialog() function of current tab controller.
        """
        try:
            self.application().currentTabController().zoomDialog()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": zoomDialog() - No tab controller found.")
   
    def saveImage(self):
        """ Calls saveImage() function of current tab controller.
        """
        try:
            self.application().currentTabController().saveImage()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": saveImage() - No tab controller found.")

    def viewClassId(self, viewClass):
        if not viewClass:# or not isinstance(viewClass, AbstractView):
            return None
        return str(viewClass) + str(viewClass.LABEL)
        
    def addCenterView(self, viewClass, default=False, enabled=True):
        """ add a View for the Center View
        
         selection: name is the menu entry, function the function to be
         added as action if the menu is selecteds
         """
        logging.debug(__name__ + ": addCenterView")
        if not issubclass(viewClass, AbstractView):
            logging.error(self.__class__.__name__ + ": addCenterView() - Cannot add non views as view. Aborting...")
            return
        if viewClass in self._availableCenterViews:
            logging.warning(self.__class__.__name__ + ": Already have a View of type "+ viewClass.__name__ +". Aborting...")
            return
        viewClassId=self.viewClassId(viewClass)
        action = self.application().createAction(viewClass.LABEL, self.switchCenterViewSlot, "Ctrl+"+ str(len(self._availableCenterViews)))
        action.setCheckable(True)
        action.setEnabled(enabled)
        if not enabled:
            self._disabledCenterViewIds = [viewClassId]
        action.setData(QVariant(viewClassId))
        if default:
            self._defaultCenterViewId = self.viewClassId(viewClass)
        self._availableCenterViews.append(viewClass)
        self._viewMenu.addAction(action)

    def switchCenterViewSlot(self):
        """ Slot for center view menu entries.
        
        It switches the tab's current center view to the selected view.
        """
        requestedViewClassId = self.sender().data().toString()
        try:
            self.application().currentTabController().switchCenterView(requestedViewClassId)
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": switchCenterViewSlot() - No tab controller found.")
        
    def openFile(self, filename=None):
        """ Open the requested file in a new tab.
        
        This method is called when the user wants to open a file with an extension this plugin.
        previously registered.
        
        This methods overwrites openFile from VispaPlugin.
        """
        logging.debug(__name__ + ": openFile " + filename)
        if self._startUp:
            self.startUp()
        if filename == None:
            return False
        base = os.path.basename(filename)
        ext = os.path.splitext(base)[1].lower().strip(".")
        if ext in [ft.extension() for ft in self.filetypes()]:
            tab = self.newTab()
            return tab.controller().open(filename)
        return False
    
    def newTab(self):
        raise NotImplementedError
        
    def boxContentAction(self):
        return self._boxContentAction

    def saveImageAction(self):
        return self._saveImageAction

    def zoomAction(self):
        return self._zoomAction

    def expandAllAction(self):
        return self._expandAllAction

    def expandToDepthAction(self):
        return self._expandToDepthAction

    def collapseAllAction(self):
        return self._collapseAllAction
