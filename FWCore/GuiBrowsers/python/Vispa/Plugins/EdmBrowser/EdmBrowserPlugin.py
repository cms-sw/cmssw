import os.path
import logging

from Vispa.Main.Application import Application
from Vispa.Plugins.EventBrowser.EventBrowserPlugin import EventBrowserPlugin
from Vispa.Views.TableView import TableView
from Vispa.Share.ThreadChain import ThreadChain
from Vispa.Main.Exceptions import NoCurrentTabControllerException,PluginIgnoredException,exception_traceback

try:
    from Vispa.Plugins.EdmBrowser.EdmDataAccessor import EdmDataAccessor
except Exception,e:
    raise PluginIgnoredException("cannot import CMSSW: " + str(e))

from Vispa.Plugins.EdmBrowser.EdmBrowserTab import EdmBrowserTab
from Vispa.Plugins.EdmBrowser.EdmBrowserBoxView import EdmBrowserBoxView
from Vispa.Plugins.EdmBrowser.EdmBrowserTabController import EdmBrowserTabController

class EdmBrowserPlugin(EventBrowserPlugin):
    """ The EdmBrowserPlugin opens edm root files in the EventBrowserTab.
    """
    
    def __init__(self, application=None, name=None):
        logging.debug(__name__ + ": __init__")
        EventBrowserPlugin.__init__(self, application)
        self.registerFiletypesFromTabController(EdmBrowserTabController)

    def startUp(self):
        EventBrowserPlugin.startUp(self)
        self.addCenterView(EdmBrowserBoxView)
        self.addCenterView(TableView)
                
    def newTab(self):
        """ Create EdmBrowserTab and add to MainWindow.
        """
        logging.debug(__name__ + ": newTab")
        tab = EdmBrowserTab(self.application().mainWindow())
        
        controller = EdmBrowserTabController(self)
        controller.setDataAccessor(EdmDataAccessor())
        tab.setController(controller)
        
        controller.boxContentDialog().addButton("&Label", "str(object.Label)")
        controller.boxContentDialog().addButton("&Type", "str(object.Type)")
        controller.boxContentDialog().addButton("&Name", "str(object.Name)")
        controller.boxContentDialog().addButton("&Pt", "str(object.Pt)")
        
        self.application().mainWindow().addTab(tab)
        return tab

    def _expandToDepth(self):
        """ Calls expandToDepthDialog() function of current tab controller.
        """
        try:
            self.application().currentTabController().expandToDepthDialog()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": _expandToDepth() - No tab controller found.")

    def _fillMenu(self):
        """ Fill specific menu.
        """
        EventBrowserPlugin._fillMenu(self)
        self._filterBranchesAction = self.application().createAction('&Filter invalid branches', self._filterBranches, "Ctrl+Shift+F")
        self._filterBranchesAction.setCheckable(True)
        self._filterBranchesAction.setChecked(True)
        self._viewMenu.addAction(self._filterBranchesAction)
        self._hideUnderscorePropertiesAction = self.application().createAction('&Hide _underscore properties', self._hideUnderscoreProperties, "Ctrl+Shift+U")
        self._hideUnderscorePropertiesAction.setCheckable(True)
        self._hideUnderscorePropertiesAction.setChecked(True)
        self._viewMenu.addAction(self._hideUnderscorePropertiesAction)
        self._viewMenu.addSeparator()
        self._eventContentAction = self.application().createAction('&Browse event content...', self._eventContent, "Ctrl+Shift+C")
        self._viewMenu.addAction(self._eventContentAction)
        self._viewMenu.addSeparator()

    def filterBranchesAction(self):
        return self._filterBranchesAction

    def _filterBranches(self):
        try:
            self.application().currentTabController().toggleFilterBranches()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": _filterBranches() - No tab controller found.")

    def eventContentAction(self):
        return self._eventContentAction

    def _eventContent(self):
        try:
            self.application().currentTabController().eventContent()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": _eventContent() - No tab controller found.")

    def hideUnderscorePropertiesAction(self):
        return self._hideUnderscorePropertiesAction

    def _hideUnderscoreProperties(self):
        try:
            self.application().currentTabController().toggleUnderscoreProperties()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": _hideUnderscoreProperties() - No tab controller found.")
