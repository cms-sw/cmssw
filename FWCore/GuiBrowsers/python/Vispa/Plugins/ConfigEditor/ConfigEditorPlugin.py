import os.path
import logging

from PyQt4.QtCore import QCoreApplication

from Vispa.Plugins.Browser.BrowserPlugin import BrowserPlugin
from Vispa.Plugins.Browser.BrowserTab import BrowserTab
from Vispa.Main.Exceptions import NoCurrentTabControllerException
from Vispa.Main.Exceptions import PluginIgnoredException

try:
    from Vispa.Plugins.ConfigEditor.ConfigDataAccessor import ConfigDataAccessor
except Exception,e:
    raise PluginIgnoredException("cannot import CMSSW: " + str(e))

from Vispa.Plugins.ConfigEditor.ConfigEditorTab import ConfigEditorTab
from Vispa.Plugins.ConfigEditor.ConfigEditorTabController import ConfigEditorTabController
from Vispa.Plugins.ConfigEditor.ConfigEditorBoxView import ConnectionStructureView
from Vispa.Plugins.ConfigEditor.ConfigEditorBoxView import SequenceStructureView
from Vispa.Views.AbstractView import NoneView

class ConfigEditorPlugin(BrowserPlugin):
    """ The ConfigEditorPlugin opens config files in a ConfigEditorTab.
    """
    
    def __init__(self, application=None, name=None):
        logging.debug(__name__ + ": __init__")
        BrowserPlugin.__init__(self, application)
        self.registerFiletypesFromTabController(ConfigEditorTabController)
        self.addNewFileAction("New configuration file", self.newFile)
        self.application().commandLineParser().add_option("-s", "--saveimage", dest="saveimage", help="save center view to image FILE", metavar="FILE")

    def startUp(self):
        BrowserPlugin.startUp(self)
        self.addCenterView(NoneView)
        self.addCenterView(ConnectionStructureView,True)
        self.addCenterView(SequenceStructureView)

    def newTab(self):
        """ Create ConfigEditorTab and add to MainWindow.
        """
        tab = ConfigEditorTab(self.application().mainWindow())
        controller = ConfigEditorTabController(self)
        controller.setDataAccessor(ConfigDataAccessor())
        tab.setController(controller)
        controller.boxContentDialog().addButton("&Type", "object.type")
        controller.boxContentDialog().addButton("&Classname", "object.classname")
        controller.boxContentDialog().addButton("&Filename", "object.filename")
        controller.boxContentDialog().addButton("&Package", "object.package")
        self.application().mainWindow().addTab(tab)
        return tab

    def newFile(self,new=True):
        """ Create ConfigEditorTab and add to MainWindow.
        """
        if self._startUp:
            self.startUp()
        if new:
            controller=self.newTab().controller()
        else:
            try:
                if isinstance(self.application().currentTabController(),ConfigEditorTabController):
                    controller=self.application().currentTabController()
            except NoCurrentTabControllerException:
                controller=None
            if not controller:
                logging.error(__name__ + ": No configuration file was opened for editing.")
                self.application().errorMessage("No configuration file was opened for editing.")
                return None
        controller.startEditMode()
