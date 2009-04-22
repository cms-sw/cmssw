import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Main.Filetype import *
    
class VispaPlugin(QObject):
    """Interface for all VispaPlugins"""
    
    def __init__(self, application = None):
        QObject.__init__(self, application)
        self._application = application
        
        self._createNewFileActions = []
        self._filetypes = []
        
    def application(self):
        """ Returns application object.
        """
        return self._application
    
    def registerFiletype(self, ext, description):
        """ Registers Filetype object for given extension with description.
        
        Description will be shown in open and save dialogs.
        """
        self._filetypes.append(Filetype(ext,description))
    
    def filetypes(self):
        """ Returns local list of Filetype objects.
        """
        return self._filetypes
    
    def registerFiletypesFromTabController(self, TabControllerClass):
        """Adds supported file types from TabControllerClass.
        
        Evaluates the static function staticSupportedFileTypes() of class TabControllerClass."""
        for (ext, description) in TabControllerClass.staticSupportedFileTypes():
            self.registerFiletype(ext, description)

    def openFile(self, filename):
        """This function has to be implemented by each plugin which can open files.
        
        On success it should return True
        """
        logging.warning('VispaPlugin: openFile() method not implemented by '+ self.__class__.__name__ +'.')
        self.application().statusBar().showMessage('Opening of desired file type not implemented.', 10000)
        return False
        
    def appendNewFileAction(self, action):
        """ Adds action to local list of new file actions.
        
        Entries in this list will appear in the main file menu.
        """
        self._createNewFileActions.append(action)
        
    def addNewFileAction(self, label, slot=None):
        """Creates a new file action with label and optionally with a callable slot set and appends it to local new file actions list. """
        self._createNewFileActions.append(self._application.createAction(label, slot,image='filenew'))
        
    def getNewFileActions(self):
        """ Returns local list of new file actions.
        """
        return self._createNewFileActions
