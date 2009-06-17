import logging

from Vispa.Main.TabController import *
from Vispa.Main.FindDialog import *
from Vispa.Main.FindAlgorithm import *

class EventContentAnalyzerTabController(TabController):
    """ """
    def __init__(self, plugin):
        logging.debug(__name__ + ": __init__")
        TabController.__init__(self, plugin)

        self._dataAccessor = None
        
        self.setFindEnabled()
        self._findDialog = None

        self._eventContentMenu = self.plugin().application().createPluginMenu('&Event content')
        configFileAction = self.plugin().application().createAction('&Add config file', self.addConfigFile, 'STRG-F')
        self._eventContentMenu.addAction(configFileAction)
        rootFileAction = self.plugin().application().createAction('&Add text file', self.addTextFile, 'STRG-T')
        self._eventContentMenu.addAction(rootFileAction)
        textFileAction = self.plugin().application().createAction('&Add root file', self.addRootFile, 'STRG-R')
        self._eventContentMenu.addAction(textFileAction)

    def setDataAccessor(self, accessor):
        """ Set the DataAccessor and show data in the TreeView.
        """
        logging.debug(__name__ + ": setDataAccessor")
        self._dataAccessor = accessor
        self.tab().centerView().setDataAccessor(self._dataAccessor)
        self.tab().centerView().updateContent()
    
    def dataAccessor(self):
        return self._dataAccessor

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
        item = self.tab().centerView().itemByObject(object)
        print object,item
        self.tab().centerView().select(item)

    def closeEvent(self, event):
        event.accept()

    def addConfigFile(self, filename=""):
        if filename=="":
            filename = str(QFileDialog.getOpenFileName(self.plugin().application().mainWindow(),
                                               'Select a config file',
                                               self.plugin().application().getLastOpenLocation(),
                                               "*.py","*.py"))
        if not filename=="":
            self._dataAccessor.addConfigFile(str(filename))
            self.tab().centerView().updateContent()
        
    def addRootFile(self, filename=""):
        if filename=="":
            filename = str(QFileDialog.getOpenFileName(self.plugin().application().mainWindow(),
                                               'Select a root file',
                                               self.plugin().application().getLastOpenLocation(),
                                               "*.root","*.root"))
        if not filename=="":
            self._dataAccessor.addRootFile(filename)
            self.tab().centerView().updateContent()

    def addTextFile(self, filename=""):
        if filename=="":
            filename = str(QFileDialog.getOpenFileName(self.plugin().application().mainWindow(),
                                               'Select a text file',
                                               self.plugin().application().getLastOpenLocation(),
                                               "*.txt","*.txt"))
        if not filename=="":
            self._dataAccessor.addTextFile(filename)
            self.tab().centerView().updateContent()

    def selected(self):
        """ Shows plugin menus when user selects tab.
        """
        logging.debug(__name__ + ": selected()")
        TabController.selected(self)
        self.plugin().application().showPluginMenu(self._eventContentMenu)

