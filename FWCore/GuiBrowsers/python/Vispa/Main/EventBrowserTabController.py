import logging

from PyQt4.QtCore import QCoreApplication

from Vispa.Main.TripleTabController import TripleTabController
from Vispa.Main.EventFileNavigator import *
from Vispa.Main.Thread import RunThread

from Vispa.Main.Workspace import *
from Vispa.Main.BoxDecayTree import *

class EventBrowserTabController(TripleTabController):
    """ The EventBrowserTabController supplies functionality for browsing objects in an EventBrowserTab and navigating through events.
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        TripleTabController.__init__(self, plugin)
        
        self.setEditable(False)

        self._viewMenu.addSeparator()
        self.addCenterView("&None", self.switchToNoneView)
        self._noneCenterView = Workspace()
        self.addCenterView("&Decay Tree Graph", self.switchToLineDecayTree, False, "Ctrl+D")
        self.addCenterView("&Box Decay Tree", self.switchToBoxDecayView, True, "Ctrl+B")
        self._boxDecayTree = BoxDecayTree()
        self.fillCenterViewSelectMenu()
        self._eventFileNavigator = EventFileNavigator(self.plugin().application())
        self.connect(self._eventFileNavigator, SIGNAL("update"), self.updateAndRestoreSelection)

    def closeEvent(self, event):
        """ Wait for EventFileNavigation to finish.
        """
        while self._eventFileNavigator.isRunning():
            QCoreApplication.instance().processEvents()
        self.dataAccessor().close()
        TripleTabController(self, event)

    def setDataAccessor(self, accessor):
        self._eventFileNavigator.setDataAccessor(accessor)
        TripleTabController.setDataAccessor(self, accessor)

    def selected(self):
        """ Show eventFileNavigator when tab is shown.
        """
        TripleTabController.selected(self)
        self._eventFileNavigator.showPluginMenu()
        if self.currentCenterView() != "&None":
            self.tab().mainWindow().application().showZoomToolBar()

    def switchToBoxDecayView(self):
        """Switches to Boxdecaytree
        """
        logging.debug(__name__ + ": switchToBoxDecayView")
        
        self.tab().setCenterView(self._boxDecayTree)
        self.tab().centerView().setUseLineDecayTree(False)
        self._boxContentAction.setVisible(True)
        if self.tab().mainWindow():
            self.tab().mainWindow().application().showZoomToolBar()
        self._saveIni()

    def switchToLineDecayTree(self):
        """ Switches to LineDecayTree .
        """
        logging.debug(__name__ + ": switchToLineDecayTree")
        self.tab().setCenterView(self._boxDecayTree)
        self._boxContentAction.setVisible(False)
        self.tab().centerView().setUseLineDecayTree(True)
        if self.tab().mainWindow():
            self.tab().mainWindow().application().showZoomToolBar()
        self._saveIni()

    def _saveIni(self):
        """ write options to ini """
        ini = self.plugin().application().ini()
        if not ini.has_section("view"):
            ini.add_section("view")
        ini.set("view", "CurrentView", self.currentCenterView())
        ini.set("view", "box content script", self._boxDecayTree.boxContentScript())
        self.plugin().application().writeIni()

    def _loadIni(self):
        """ read options from ini """
        logging.debug(__name__ + ": Loading Ini")
        ini = self.plugin().application().ini()
        if ini.has_option("view", "CurrentView"):
          proposed_view = ini.get("view", "CurrentView")
          if self._availableCenterViews.has_key(proposed_view):
            self._centerViewActions[self._currentCenterView].setChecked(False)
            self._currentCenterView = proposed_view
            self._centerViewActions[self._currentCenterView].setChecked(True)
        self._availableCenterViews[self._currentCenterView]()
        if ini.has_option("view", "box content script"):
            self._boxDecayTree.setBoxContentScript(str(ini.get("view", "box content script")))
            self._boxContentDialog.setScript(str(ini.get("view", "box content script")))
        self.connect(self.tab().centerView(), SIGNAL("widgetSelected"), self.onWidgetSelected)

    def switchToNoneView(self):
        logging.debug(__name__ + ": switchToNoneView")
        self.tab().setCenterView(self._noneCenterView)
        if self.tab().mainWindow():
            self.tab().mainWindow().application().hideZoomToolBar()
        self._saveIni()

    def setTab(self, tab):
        TripleTabController.setTab(self, tab)
        self._loadIni()

    def refresh(self):
        statusMessage = self.plugin().application().startWorking("Reopening file")
        self.dataAccessor().close()
        self.readFile(self._filename)
        self.plugin().application().stopWorking(statusMessage)
            
    def readFile(self, filename):
        """ Reads in the file in a separate thread.
        """
        thread = RunThread(self.dataAccessor().open, filename)
        while thread.isRunning():
            QCoreApplication.instance().processEvents()
        if thread.returnValue:
            self.updateAndRestoreSelection()
            return True
        return False
