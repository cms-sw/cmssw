import os.path
import logging

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Vispa.Main.VispaPlugin import *
from Vispa.Main.Thread import *
from ConfigBrowserTab import *
from ConfigBrowserTabController import *
from ConfigDataAccessor import *

class ConfigBrowserPlugin(VispaPlugin):
    """ The ConfigBrowserPlugin opens config files in a ConfigBrowserTab.
    """
    
    def __init__(self, application=None, name=None):
        logging.debug(__name__ + ": __init__")
        VispaPlugin.__init__(self, application)
        self.registerFiletypesFromTabController(ConfigBrowserTabController)
        self.application().commandLineParser().add_option("-s", "--saveimage", dest="saveimage", help="save center view to FILE (*.bmp,*.png)", metavar="FILE")

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
            accessor = ConfigDataAccessor()
            openSucessfully = False
            thread = RunThread(accessor.open, filename)
            while thread.isRunning():
                qApp.processEvents()
            if thread.returnValue:
                tab, controller = self.newTab()
                controller.setFilename(filename)
                controller.updateLabel()
                if not accessor.process():
                    controller.dumpAction().setVisible(False)
                    controller.readOnlyAction().setChecked(True)
                    controller.readOnlyAction().setEnabled(False)
                    controller.dotExportAction().setEnabled(False)
                    controller.readOnlyMode()
                    self.application().warningMessage("Config does not contain a process and is opened in read-only mode.")
                controller.setDataAccessor(accessor)
                if self.application().commandLineOptions().saveimage:
                    tab.centerView().updateConnections()
                    controller.saveImage(self.application().commandLineOptions().saveimage)
                    print "Saved image to", self.application().commandLineOptions().saveimage, "."
                    sys.exit(2)
                return True
        return False

    def newTab(self):
        """ Create ConfigBrowserTab and add to MainWindow.
        """
        tab = ConfigBrowserTab(self.application().mainWindow())
        controller = ConfigBrowserTabController(self)
        tab.setController(controller)
        controller.boxContentDialog().addButton("&Label", "str(object.label)")
        controller.boxContentDialog().addButton("&Type", "str(object.type)")
        controller.boxContentDialog().addButton("&Classname", "str(object.classname)")
        controller.boxContentDialog().addButton("&Filename", "str(object.filename)")
        controller.boxContentDialog().addButton("&Package", "str(object.package)")
        self.application().mainWindow().addTab(tab)
        return tab, controller
