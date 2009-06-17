import os.path
import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Vispa.Main.VispaPlugin import *
from EventContentAnalyzerTab import *
from EventContentAnalyzerTabController import *
from EventContentDataAccessor import *

class EventContentAnalyzerPlugin(VispaPlugin):
    """ The EventContentAnalyzerPlugin compares the file content of edm root files and input and output of configuration files.
    """
    
    def __init__(self, application=None, name=None):
        logging.debug(__name__ + ": __init__")
        VispaPlugin.__init__(self, application)
        self.addNewFileAction("&New Event Content Analyzer", self.newTab)

    def newTab(self):
        """ Create EventContentAnalyzerTab and add to MainWindow.
        """
        tab = EventContentAnalyzerTab(self.application().mainWindow())
        controller = EventContentAnalyzerTabController(self)
        tab.setController(controller)
        controller.setDataAccessor(EventContentDataAccessor())
        controller.setEditable(False)
        self.application().mainWindow().addTab(tab)
        return tab, controller
