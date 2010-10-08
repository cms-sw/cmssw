import logging
import os.path
import sys

from PyQt4.QtCore import QCoreApplication,SIGNAL
from PyQt4.QtGui import QInputDialog

from Vispa.Main.Application import Application
from Vispa.Plugins.Browser.BrowserTabController import BrowserTabController
from Vispa.Plugins.EventBrowser.EventFileAccessor import EventFileAccessor
from Vispa.Share.ThreadChain import ThreadChain

class EventBrowserTabController(BrowserTabController):
    """ The EventBrowserTabController supplies functionality for browsing objects in an EventBrowserTab and navigating through events.
    """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        BrowserTabController.__init__(self, plugin)
        
        self.setEditable(False)
        self._thread=None
        self._navigateTo=None
        self.connect(self,SIGNAL("navigate"),self.navigate)
        
    def close(self):
        """ Close data file.
        """
        result=BrowserTabController.close(self)
        if result:
            self.dataAccessor().close()
        return result

    def setDataAccessor(self, accessor):
        if not isinstance(accessor, EventFileAccessor):
            raise TypeError(__name__ + " requires data accessor of type EventFileAccessor.")
        BrowserTabController.setDataAccessor(self, accessor)

    def activated(self):
        """ Show event menu when tab is shown.
        """
        BrowserTabController.activated(self)
        self.updateEventNumberDisplay()
        if not self.isEditable():
            self.plugin().application().showPluginMenu(self.plugin().navigateMenu())
            self.plugin().application().showPluginToolBar(self.plugin().navigateToolBar())
    
    def refresh(self):
        eventNum=self._dataAccessor.eventNumber()
        self._fileModifcationTimestamp = os.path.getmtime(self._filename)
        self.navigate(0)
        self.navigate(eventNum)
            
    def readFile(self, filename):
        """ Reads in the file in a separate thread.
        """
        self.cancel()
        self._thread = ThreadChain(self.dataAccessor().open, filename)
        while self._thread.isRunning():
            if not Application.NO_PROCESS_EVENTS:
                QCoreApplication.instance().processEvents()
        if self._thread.returnValue():
            self.updateEventNumberDisplay()
            return True
        return False

    def isBusy(self):
        return BrowserTabController.isBusy(self) or\
               (self._thread and self._thread.isRunning())

    def navigate(self,to):
        # remember if navigation is ongoing
        navigating=self._navigateTo
        # set where to navigate
        self._navigateTo=to
        # if navigation is ongoing return
        if navigating!=None:
            return
        # if window is busy navigate later
        if self.isBusy():
            self.emit(SIGNAL("navigate"),to)
            return
        update=False
        while self._navigateTo!=None:
            current=self._navigateTo
            if self._navigateTo==0: 
                statusMessage = self.plugin().application().startWorking("Reopening file")
                self.dataAccessor().close()
                self.readFile(self._filename)
            else:
                statusMessage = self.plugin().application().startWorking("Navigate in file")
                if self._dataAccessor.goto(self._navigateTo):
                    update=True
            if current==self._navigateTo:
                self._navigateTo=None
                if update:
                    self.updateContent()
            self.updateEventNumberDisplay()
            self.plugin().application().stopWorking(statusMessage)

    def first(self):
        """ Navigate and to first event.
        """
        logging.debug(__name__ + ": first")
        self.cancel()
        currentEvent=self.dataAccessor().eventNumber()
        if currentEvent>1:
            self.navigate(1)

    def previous(self):
        """ Navigate and to previous event.
        """
        logging.debug(__name__ + ": previous")
        self.cancel()
        currentEvent=self.dataAccessor().eventNumber()
        if currentEvent>1:
            self.navigate(currentEvent-1)

    def next(self):
        """ Navigate and to next event.
        """
        logging.debug(__name__ + ": next")
        self.cancel()
        currentEvent=self.dataAccessor().eventNumber()
        allEvents=self.dataAccessor().numberOfEvents()
        if allEvents==None:
            allEvents=sys.maxint
        if currentEvent<allEvents:
            self.navigate(currentEvent+1)

    def last(self):
        """ Navigate and to last event.
        """
        logging.debug(__name__ + ": last")
        self.cancel()
        currentEvent=self.dataAccessor().eventNumber()
        allEvents=self.dataAccessor().numberOfEvents()
        if allEvents==None:
            allEvents=sys.maxint
        if currentEvent<allEvents:
            self.navigate(allEvents)

    def goto(self, number=None):
        """ Ask event number in dialog and navigate and to event.
        """
        logging.debug(__name__ + ": goto")
        if self._dataAccessor.numberOfEvents():
            max = self._dataAccessor.numberOfEvents()
        else:
            max = sys.maxint
        if number!=None:
            ok=(number>=1, number<=max)
        else:
            if hasattr(QInputDialog, "getInteger"):
                # Qt 4.3
                (number, ok) = QInputDialog.getInteger(self.plugin().application().mainWindow(), "Goto...", "Enter event number:", self._dataAccessor.eventNumber(), 1, max)
            else:
                # Qt 4.5
                (number, ok) = QInputDialog.getInt(self.plugin().application().mainWindow(), "Goto...", "Enter event number:", self._dataAccessor.eventNumber(), 1, max)
        if ok:
            self.cancel()
            currentEvent=self.dataAccessor().eventNumber()
            if currentEvent!=number:
                self.navigate(number)

    def updateEventNumberDisplay(self):
        eventDisplayString = str(self._dataAccessor.eventNumber()) + "/"
        if self._dataAccessor.numberOfEvents():
            eventDisplayString += str(self._dataAccessor.numberOfEvents())
        else:
            eventDisplayString += "?"
        self.plugin().eventNumberDisplay().setText(eventDisplayString)
