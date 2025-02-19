import os.path
import logging

from PyQt4.QtGui import QLabel

from Vispa.Plugins.Browser.BrowserPlugin import BrowserPlugin

from Vispa.Main.Exceptions import NoCurrentTabControllerException

class EventBrowserPlugin(BrowserPlugin):
    """ The EventBrowserPlugin supports navigation in files.
    """
    
    def __init__(self, application=None):
        logging.debug(__name__ + ": __init__")
        BrowserPlugin.__init__(self, application)
    
    def startUp(self):
        BrowserPlugin.startUp(self)
        self._fillEventNavigatorMenu()

    def _fillEventNavigatorMenu(self):
        """ Fill EventNavigator specific menu.
        """
        self._navigateMenu = self._application.createPluginMenu('&Navigate')
        self._navigateToolBar = self._application.createPluginToolBar('&Navigate')
        self._firstAction = self._application.createAction('&First', self.first, "Ctrl+Home", "first")
        self._previousAction = self._application.createAction('&Previous', self.previous, "Ctrl+PgUp", "previous")
        self._nextAction = self._application.createAction('&Next', self.next, "Ctrl+PgDown", "next")
        self._lastAction = self._application.createAction('&Last', self.last, "Ctrl+End", "last")
        self._gotoAction = self._application.createAction('&Goto...', self.goto, "Ctrl+G")
        self._eventNumberDisplay = QLabel("")
        self._navigateMenu.addAction(self._firstAction)
        self._navigateMenu.addAction(self._previousAction)
        self._navigateMenu.addAction(self._nextAction)
        self._navigateMenu.addAction(self._lastAction)
        self._navigateMenu.addAction(self._gotoAction)
        self._navigateToolBar.addAction(self._firstAction)
        self._navigateToolBar.addAction(self._previousAction)
        self._navigateToolBar.addWidget(self._eventNumberDisplay)
        self._navigateToolBar.addAction(self._nextAction)
        self._navigateToolBar.addAction(self._lastAction)

    def eventNumberDisplay(self):
        return self._eventNumberDisplay

    def first(self):
        """ Calls first() function of current tab controller.
        """
        try:
            self.application().currentTabController().first()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": first() - No tab controller found.")

    def previous(self):
        """ Calls previous() function of current tab controller.
        """
        try:
            self.application().currentTabController().previous()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": previous() - No tab controller found.")

    def next(self):
        """ Calls next() function of current tab controller.
        """
        try:
            self.application().currentTabController().next()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": next() - No tab controller found.")

    def last(self):
        """ Calls last() function of current tab controller.
        """
        try:
            self.application().currentTabController().last()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": last() - No tab controller found.")

    def goto(self):
        """ Calls goto() function of current tab controller.
        """
        try:
            self.application().currentTabController().goto()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": goto() - No tab controller found.")

    def navigateMenu(self):
        return self._navigateMenu

    def navigateToolBar(self):
        return self._navigateToolBar
    