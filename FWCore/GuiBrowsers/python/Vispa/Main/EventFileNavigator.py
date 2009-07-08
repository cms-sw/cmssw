import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Thread import *
from EventFileAccessor import *

class EventFileNavigator(QObject):
    """ The EventFileNavigator creates a menu and toolbar entries for navigating through events using an EventFileAccessor.
    
    Event navigation if performed in a separate thread.
    """
    def __init__(self, application, parent=None):
        QObject.__init__(self, parent)
        logging.debug(__name__ + ": __init__")
        self._application = application
        self._dataAccessor = None
        self._statusMessage = None
        self._fillEventNavigatorMenu()
        self._threadChain = ThreadChain()
        self.connect(self._threadChain, SIGNAL("finishedThreadChain"), self.finishedThreadChain)
        
    def setDataAccessor(self, accessor):
        """ Set the DataAccessor and update event number display.
        """
        logging.debug(__name__ + ": setDataAccessor")
        if not isinstance(accessor, EventFileAccessor):
            raise TypeError(__name__ + " requires data accessor of type EventFileAccessor.")
        self._dataAccessor = accessor
    
    def dataAccessor(self):
        return self._dataAccessor

    def _fillEventNavigatorMenu(self):
        """ Fill EventNavigator specific menu.
        """
        self._navigateMenu = self._application.createPluginMenu('&Navigate')
        self._navigateToolBar = self._application.createPluginToolBar('&Navigate')
        firstAction = self._application.createAction('&First', self.first, "Ctrl+Home", "first")
        previousAction = self._application.createAction('&Previous', self.previous, "Ctrl+PgUp", "previous")
        nextAction = self._application.createAction('&Next', self.next, "Ctrl+PgDown", "next")
        lastAction = self._application.createAction('&Last', self.last, "Ctrl+End", "last")
        gotoAction = self._application.createAction('&Goto...', self.goto, "Ctrl+G")
        self._eventNumberDisplay = QLabel("")
        self._navigateMenu.addAction(firstAction)
        self._navigateMenu.addAction(previousAction)
        self._navigateMenu.addAction(nextAction)
        self._navigateMenu.addAction(lastAction)
        self._navigateMenu.addAction(gotoAction)
        self._navigateToolBar.addAction(firstAction)
        self._navigateToolBar.addAction(previousAction)
        self._navigateToolBar.addWidget(self._eventNumberDisplay)
        self._navigateToolBar.addAction(nextAction)
        self._navigateToolBar.addAction(lastAction)

    def _setEventNumberDisplay(self, label):  
        self._eventNumberDisplay.setText(label)
    
    def first(self):
        """ Navigate and to first event.
        """
        logging.debug(__name__ + ": first")
        if self._statusMessage != None: 
            self._statusMessage = self._application.startWorking("Navigate in file")
        self._threadChain.addCommand(self._dataAccessor.first)
        if not self._threadChain.isRunning():
            self._threadChain.start()

    def previous(self):
        """ Navigate and to previous event.
        """
        logging.debug(__name__ + ": previous")
        if self._statusMessage != None: 
            self._statusMessage = self._application.startWorking("Navigate in file")
        self._threadChain.addCommand(self._dataAccessor.previous)
        if not self._threadChain.isRunning():
            self._threadChain.start()

    def next(self):
        """ Navigate and to next event.
        """
        logging.debug(__name__ + ": next")
        if self._statusMessage != None: 
            self._statusMessage = self._application.startWorking("Navigate in file")
        self._threadChain.addCommand(self._dataAccessor.next)
        if not self._threadChain.isRunning():
            self._threadChain.start()

    def last(self):
        """ Navigate and to last event.
        """
        if self._statusMessage != None: 
            self._statusMessage = self._application.startWorking("Navigate in file")
        self._threadChain.addCommand(self._dataAccessor.last)
        if not self._threadChain.isRunning():
            self._threadChain.start()

    def goto(self, number=None):
        """ Ask event number in dialog and navigate and to event.
        """
        if self._dataAccessor.numberOfEvents():
            max = self._dataAccessor.numberOfEvents()
        else:
            max = 1000000000
        if number!=None:
            ok=(number>=1, number<=max)
        else:
            if hasattr(QInputDialog, "getInteger"):
                # Qt 4.3
                (number, ok) = QInputDialog.getInteger(self._application.mainWindow(), "Goto...", "Enter event number:", self._dataAccessor.eventNumber(), 1, max)
            else:
                # Qt 4.5
                (number, ok) = QInputDialog.getInt(self._application.mainWindow(), "Goto...", "Enter event number:", self._dataAccessor.eventNumber(), 1, max)
        if ok:
            if self._statusMessage != None: 
                self._statusMessage = self._application.startWorking("Navigate in file")
            self._threadChain.addCommand(self._dataAccessor.goto, number)
            if not self._threadChain.isRunning():
                self._threadChain.start()

    def updateEventNumberDisplay(self):
        eventDisplayString = str(self._dataAccessor.eventNumber()) + "/"
        if self._dataAccessor.numberOfEvents():
            eventDisplayString += str(self._dataAccessor.numberOfEvents())
        else:
            eventDisplayString += "?"
        self._setEventNumberDisplay(eventDisplayString)

    def finishedThreadChain(self, results):
        """ Send an update signal when when the navigation operation is done and successful.
        
        A controller can connect to this signal to update the display.
        """
        logging.debug(__name__ + ": finishedThreadChain")
        self.updateEventNumberDisplay()
        if True in results:
            self.emit(SIGNAL("update"))
        if self._statusMessage != None:
            self._application.stopWorking(self._statusMessage)
        self._statusMessage = None

    def isRunning(self):
        """ Navigation threads still running?
        """
        return self._threadChain.isRunning()

    def showPluginMenu(self):
        """ Call this when a tab is selected, to show the navigation menu.
        """
        self._application.showPluginMenu(self._navigateMenu)
        self._application.showPluginToolBar(self._navigateToolBar)
