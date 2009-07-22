import logging

from PyQt4.QtCore import QCoreApplication

from Vispa.Main.BrowserController import BrowserController
from Vispa.Main.EventFileNavigator import *
from Vispa.Main.Thread import RunThread

from Vispa.Main.WidgetView import WidgetView
from Vispa.Main.BoxDecayTree import *
from Vispa.Main.LineDecayTree import *

#import Vispa.Main.Preferences as Preferences

#if Preferences.get('RootCanvasViewEnabled'):
#  try:
#    from Vispa.Main.RootCanvasView import RootCanvasView
#    RootCanvasViewEnable = True
#  except:
#    RootCanvasViewEnable = False
#    logging.error(__name__ + ": RootCanvasView not availeable")
#else:
#  RootCanvasViewEnable = False
try:
  from Vispa.Main.RootCanvasView import RootCanvasView
  RootCanvasViewEnable = True
except:
  RootCanvasViewEnable = False

class EventBrowserTabController(BrowserController):
    """ The EventBrowserTabController supplies functionality for browsing objects in an EventBrowserTab and navigating through events.
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        BrowserController.__init__(self, plugin)
        
        self.setEditable(False)

        self._viewMenu.addSeparator()
        self.addCenterView("&None", self.switchToNoneView, False, "Ctrl+0")
        self.addCenterView("&Box Decay Tree", self.switchToBoxDecayView, True, "Ctrl+1")
        self.addCenterView("&Decay Tree Graph", self.switchToLineDecayTree, False, "Ctrl+2")
        self.addCenterView("&Root Canvas View ", self.switchToRootCanvasView, False, "Ctrl+3",enabled=RootCanvasViewEnable)
        
        self._centerView = None
        #self._boxDecayTree = BoxDecayTree()
        self.fillCenterViewSelectMenu()
        self._eventFileNavigator = EventFileNavigator(self.plugin().application())
        self.connect(self._eventFileNavigator, SIGNAL("update"), self.updateContent)

    def closeEvent(self, event):
        """ Wait for EventFileNavigation to finish.
        """
        while self._eventFileNavigator.isRunning():
            QCoreApplication.instance().processEvents()
        self.dataAccessor().close()
        BrowserController(self, event)

    def setDataAccessor(self, accessor):
        self._eventFileNavigator.setDataAccessor(accessor)
        BrowserController.setDataAccessor(self, accessor)

    def selected(self):
        """ Show eventFileNavigator when tab is shown.
        """
        BrowserController.selected(self)
        self._eventFileNavigator.showPluginMenu()
        if self.currentCenterView() != "&None":
            self.tab().mainWindow().application().showZoomToolBar()
   
    def switchToNoneView(self):
        logging.debug(__name__ + ": switchToNoneView")
        self._centerView = WidgetView()
        self.tab().setCenterView(self._centerView)
        if self.tab().mainWindow():
            self.tab().mainWindow().application().hideZoomToolBar()
        self._saveIni()

    def switchToBoxDecayView(self):
        """Switches to Boxdecaytree
        """
        logging.debug(__name__ + ": switchToBoxDecayView")
       
        if not isinstance(self._centerView, BoxDecayTree):
          self._centerView = BoxDecayTree()
        self.tab().setCenterView(self._centerView)
        self.tab().centerView().setSubView(None)
        self._boxContentAction.setVisible(True)
        if self.tab().mainWindow():
            self.tab().mainWindow().application().showZoomToolBar()
        self._saveIni()

    def switchToLineDecayTree(self):
        """ Switches to LineDecayTree .
        """
        logging.debug(__name__ + ": switchToLineDecayTree")
        if not isinstance(self._centerView, BoxDecayTree):
          self._centerView = BoxDecayTree()
        self.tab().setCenterView(self._centerView)
        self.tab().centerView().setSubView(LineDecayTree)
        self._boxContentAction.setVisible(False)
        if self.tab().mainWindow():
            self.tab().mainWindow().application().showZoomToolBar()
        self._saveIni()

    def switchToRootCanvasView(self):
        """Switches to RootCanvasView
        """
        logging.debug(__name__ + ": switchToRootCanvasView")
         
        self.tab().setCenterView(RootCanvasView())
        #self.tab().centerView().setSubView(None)
        self._boxContentAction.setVisible(False)
        if self.tab().mainWindow():
            self.tab().mainWindow().application().showZoomToolBar()
        self._saveIni()


    def _saveIni(self):
        """ write options to ini """
        ini = self.plugin().application().ini()
        if not ini.has_section("view"):
            ini.add_section("view")
        ini.set("view", "CurrentView", self.currentCenterView())
        if isinstance(self._centerView, BoxDecayTree):
          ini.set("view", "box content script", self._centerView.boxContentScript())
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
        self.connect(self.tab().centerView(), SIGNAL("selected"), self.onSelected)
        if ini.has_option("view", "box content script") and isinstance(self._centerView, BoxDecayTree):
            self._centerView.setBoxContentScript(str(ini.get("view", "box content script")))
            self._boxContentDialog.setScript(str(ini.get("view", "box content script")))

   
    def setTab(self, tab):
        BrowserController.setTab(self, tab)
        self._loadIni()

    def refresh(self):
        self.dataAccessor().close()
        BrowserController.refresh(self)
            
    def readFile(self, filename):
        """ Reads in the file in a separate thread.
        """
        thread = RunThread(self.dataAccessor().open, filename)
        while thread.isRunning():
            QCoreApplication.instance().processEvents()
        if thread.returnValue:
            self._eventFileNavigator.updateEventNumberDisplay()
            return True
        return False
