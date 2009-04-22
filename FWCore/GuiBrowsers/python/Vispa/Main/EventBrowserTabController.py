import logging

from Vispa.Main.TripleTabController import *
from Vispa.Main.EventFileNavigator import *

from Vispa.Main.NoneCenterView import *
from Vispa.Main.BoxDecayTree import *

class EventBrowserTabController(TripleTabController):
    """ The EventBrowserTabController supplies functionality for browsing objects in an EventBrowserTab and navigating through events.
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        TripleTabController.__init__(self, plugin)
        
        self.setEditable(False)

        self._viewMenu.addSeparator()
        self.addCenterView("None", self.switchToNoneView)
        self.addCenterView("Decay Tree Graph", self.switchToLineDecayTree)
        self.addCenterView("Box Decay Tree", self.switchToBoxDecayView,True)
        self.fillCenterViewSelectMenu()
        self._eventFileNavigator = EventFileNavigator(self.plugin().application())
        self.connect(self._eventFileNavigator, SIGNAL("update"), self.updateAndRestoreSelection)
        
    def closeEvent(self, event):
        """ Wait for EventFileNavigation to finish.
        """
        while self._eventFileNavigator.isRunning():
            qApp.processEvents()
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

    def switchToBoxDecayView(self):
        """Switches to Boxdecaytree
        """
        logging.debug(__name__ + ": switchToBoxDecayView")
        
        self.tab().setCenterView(BoxDecayTree())
        self.tab().centerView().setUseLineDecayTree(False)
        self._boxContentAction.setVisible(True)
        self._saveIni()

    def switchToLineDecayTree(self):
        """ Switches to LineDecayTree .
        """
        logging.debug(__name__ + ": switchToLineDecayTree")
        self.tab().setCenterView(BoxDecayTree())
        self._boxContentAction.setVisible(False)
        self.tab().centerView().setUseLineDecayTree(True)
        self._saveIni()

    def _saveIni(self):
        """ write options to ini """
        ini = self.plugin().application().ini()
        if not ini.has_section("view"):
            ini.add_section("view")
        ini.set("view", "CurrentView", self.getCurrentCenterView())
        self.plugin().application().writeIni()

    def _loadIni(self):
        """ read options from ini """
        logging.debug(__name__ + ": Loading Ini")
        print '******************'
        ini = self.plugin().application().ini()
        if ini.has_option("view","CurrentView"):
          proposed_view = ini.get("view","CurrentView")
          if self._availableCenterViews.has_key(proposed_view):
            self._centerViewActions[self._currentCenterView].setChecked(False)
            self._currentCenterView = proposed_view
            self._centerViewActions[self._currentCenterView].setChecked(True)
        
        self._availableCenterViews[self._currentCenterView]()
        self.connect(self.tab().centerView(), SIGNAL("widgetSelected"), self.onWidgetSelected)
        #reconnect data accessors and stuff
        self.tab().centerView().setDataAccessor(self._dataAccessor)
        self.tab().centerView().updateContent()
        self.updateAndRestoreSelection() 

    def switchToNoneView(self):
        logging.debug(__name__ + ": switchToNoneView")
        self.tab().setCenterView(NoneCenterView())
        self._saveIni()

