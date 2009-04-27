import os.path
import logging

from PyQt4.QtCore import QCoreApplication

from Vispa.Main.VispaPlugin import VispaPlugin
from Vispa.Main.Thread import RunThread
from ConfigBrowserTab import ConfigBrowserTab
from ConfigBrowserTabController import ConfigBrowserTabController
from ConfigDataAccessor import ConfigDataAccessor

class ConfigBrowserPlugin(VispaPlugin):
    """ The ConfigBrowserPlugin opens config files in a ConfigBrowserTab.
    """
    
    def __init__(self, application=None, name=None):
        logging.debug(__name__ + ": __init__")
        VispaPlugin.__init__(self, application)
        self.registerFiletypesFromTabController(ConfigBrowserTabController)
        self.application().commandLineParser().add_option("-s", "--saveimage", dest="saveimage", help="save center view to image FILE", metavar="FILE")

    def openFile(self, filename=None):
        """Open the requested file in a new tab.
        
        This method is called when the user wants to open a file with an extension this plugin.
        previously registered.
        
        This methods overwrites openFile from VispaPlugin.
        """
        logging.debug(__name__ + ": openFile " + filename)
        if filename == None:
            return False
        base = os.path.basename(filename)
        ext = os.path.splitext(base)[1].lower().strip(".")
        if ext in [ft.extension for ft in self.filetypes()]:
            tab, controller = self.newTab()
            accessor = ConfigDataAccessor()
            controller.setDataAccessor(accessor)
            return controller.open(filename)
        return False

    def newTab(self):
        """ Create ConfigBrowserTab and add to MainWindow.
        """
        tab = ConfigBrowserTab(self.application().mainWindow())
        controller = ConfigBrowserTabController(self)
        tab.setController(controller)
        controller.boxContentDialog().addButton("&Label", "object.label")
        controller.boxContentDialog().addButton("&Type", "object.type")
        controller.boxContentDialog().addButton("&Classname", "object.classname")
        controller.boxContentDialog().addButton("&Filename", "object.filename")
        controller.boxContentDialog().addButton("&Package", "object.package")
        self.application().mainWindow().addTab(tab)
        return tab, controller
