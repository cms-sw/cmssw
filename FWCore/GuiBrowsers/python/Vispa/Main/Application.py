import os
import sys
import string
import commands
import platform
import logging
import logging.handlers
import ConfigParser
import optparse
import webbrowser
import subprocess

from PyQt4.QtCore import SIGNAL,qVersion,QString,QVariant, Qt
from PyQt4.QtGui import QApplication,QMenu,QPixmap,QAction,QFileDialog,QIcon,QMessageBox

from Vispa.Main.Directories import logDirectory,pluginDirectory,baseDirectory,homeDirectory,iniFileName,applicationName,docDirectory,websiteUrl
from Vispa.Main.MainWindow import MainWindow
from Vispa.Main.AbstractTab import AbstractTab
from Vispa.Main.Filetype import Filetype
from Vispa.Main.Exceptions import *
from Vispa.Main.AboutDialog import AboutDialog
from Vispa.Main.RotatingIcon import RotatingIcon 
#from PreferencesEditor import PreferencesEditor
import Vispa.__init__

import Resources

class Application(QApplication):

    MAX_RECENT_FILES = 30
    MAX_VISIBLE_RECENT_FILES = 10
    MAX_VISIBLE_UNDO_EVENTS = 10
    FAILED_LOADING_PLUGINS_ERROR = "Errors while loading plugins. For details see error output or log file.\n\nThe following plugins won't work correctly:\n\n"
    TAB_PREMATURELY_CLOSED_WARNING = "Tab was closed before user request could be handled."
    NO_PROCESS_EVENTS = False

    def __init__(self, argv):
        QApplication.__init__(self, argv)
        self._version = None
        self._plugins = []
        self._closeAllFlag = False
        self._knownFiltersList = []
        self._knownExtensionsDictionary = {}
        self._pluginMenus = []
        self._pluginToolBars = []
        self._recentFiles = []
        self._ini = None
        self._iniFileName = iniFileName
        self._zoomToolBar = None
        self._undoToolBar = None
        self._messageId=0
        self._loadablePlugins = {}
        self._logFile = None
        
        self._initLogging()

        logging.debug('Running with Qt-Version ' + str(qVersion()))

        self._initCommandLineAttributes()

        self._loadIni()

        self.setVersion(Vispa.__init__.__version__)

        self._window = MainWindow(self, applicationName)
        self._window.show()
                
        self._loadPlugins()
        
        self._fillFileMenu()
        self._fillEditMenu()
        self._fillHelpMenu()
        
        self.createUndoToolBar()
        self.createZoomToolBar()
        self.hidePluginMenus()
        self.hidePluginToolBars()
        self.createStatusBar()
        self.updateMenu()
    
        self._readCommandLineAttributes()
        self._connectSignals()

    def commandLineParser(self):
        return self._commandLineParser
        
    def commandLineOptions(self):
        return self._commandLineOptions
        
    def setVersion(self, version):
        self._version = version
        
    def version(self):
        """ Returns version string.
        """
        return self._version
        
    def atLeastQtVersion(self, versionString):
        """ Returns True if given versionString is newer than current used version of Qt.
        """
        [majorV, minorV, revisionV] = versionString.split(".")
        [majorQ, minorQ, revisionQ] = str(qVersion()).split(".")
        if majorV > majorQ:
            return True
        elif majorV < majorQ:
            return False
        elif majorV == majorQ:
            if minorV > minorQ:
                return True
            elif minorV < minorQ:
                return False
            elif minorV == minorQ:
                if revisionV > revisionQ:
                    return True
                elif revisionV < revisionQ:
                    return False
                elif revisionV == revisionQ:
                    return True
        return False

    def _setCommandLineOptions(self):
        """ Set the available command line options.
        """
        self._commandLineParser.add_option("-f", "--file", dest="filename", help="open a FILE", metavar="FILE")
        self._commandLineParser.add_option("-l", "--loglevel", dest="loglevel", help="set LOGLEVEL to 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL", metavar="LOGLEVEL", type="int")

    def _initCommandLineAttributes(self):
        """ Initialize command line parser.
        
        After calling this function, plugins may add options.
        """
        class QuiteOptionParser(optparse.OptionParser):
            def __init__(self):
                optparse.OptionParser.__init__(self,add_help_option=False)
            def error(self,message=""):
                pass
        self._commandLineParser = QuiteOptionParser()
        self._setCommandLineOptions()
        (self._commandLineOptions, self._args) = self._commandLineParser.parse_args()
        if self._commandLineOptions.loglevel:
            logging.root.setLevel(self._commandLineOptions.loglevel)
        self._commandLineParser = optparse.OptionParser()
        self._setCommandLineOptions()
        
    def _readCommandLineAttributes(self):
        """ Analyzes the command line attributes and print usage summary if required.
        """
        (self._commandLineOptions, self._args) = self._commandLineParser.parse_args()
        if self._commandLineOptions.filename:
            self.mainWindow().setStartupScreenVisible(False)
            self.openFile(self._commandLineOptions.filename)
        if len(self._args) > 0:
            self.mainWindow().setStartupScreenVisible(False)
            self.openFile(self._args[0])
        
    def _checkFile(self, filename):
        """ Check if logfile is closed correctly
        """
        finished = True
        file = open(filename, "r")
        for line in file.readlines():
            if "INFO Start logging" in line:
                finished = False
            if "INFO Stop logging" in line:
                finished = True
        return finished

    def _initLogging(self):
        """ Add logging handlers for a log file as well as stderr.
        """ 
        instance = 0
        done = False
        while not done:
            # iterate name of log file for several instances of vispa
            instance += 1
            logfile = os.path.join(logDirectory, "log" + str(instance) + ".txt")
            # do not create more than 10 files
            if instance > 10:
                instance = 1
                logfile = os.path.join(logDirectory, "log" + str(instance) + ".txt")
                done = True
                break
            if not os.path.exists(logfile):
                done = True
                break
            done = self._checkFile(logfile)
        
        # clean up old logs
        nextlogfile = os.path.join(logDirectory, "log" + str(instance + 1) + ".txt")
        if os.path.exists(nextlogfile):
            if not self._checkFile(nextlogfile):
                file = open(nextlogfile, "a")
                file.write("Cleaning up logfile after abnormal termination: INFO Stop logging\n")

        if os.path.exists(logDirectory):
            handler1 = logging.handlers.RotatingFileHandler(logfile, maxBytes=100000, backupCount=1)
            formatter1 = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler1.setFormatter(formatter1)
            self._logFile = logfile

        handler2 = logging.StreamHandler(sys.stderr)
        formatter2 = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler2.setFormatter(formatter2)
        
        logging.root.handlers = []
        if os.path.exists(logDirectory):
            logging.root.addHandler(handler1)
        logging.root.addHandler(handler2)
        #logging.root.setLevel(logging.INFO)

        self._infologger = logging.getLogger("info")
        self._infologger.setLevel(logging.INFO)
        self._infologger.handlers = []
        if self._logFile:
            self._infologger.info("Start logging to " + self._logFile)
        
    def run(self):
        """ Show the MainWindow and run the application.
        """
        #logging.debug('Application: run()')

        self.exec_()
        if self._logFile:
            self._infologger.info("Stop logging to " + self._logFile)

    def _connectSignals(self):
        """ Connect signal to observe the TabWidget in the MainWindow.
        """
        logging.debug('Application: _connectSignals()')
        self.connect(self._window.tabWidget(), SIGNAL("currentChanged(int)"), self.tabChanged)
        self.connect(self._window, SIGNAL("windowActivated()"), self.tabChanged)
        self.connect(self._window.tabWidget(), SIGNAL("tabCloseRequested(int)"), self.tabCloseRequest)
        
    def _loadPlugins(self):
        """ Search all subfolders of the plugin directory for vispa plugins and registers them.
        """
        logging.debug('Application: _loadPlugins()')
        dirs = ["Vispa.Plugins." + str(f) for f in os.listdir(pluginDirectory)
                if os.path.isdir(os.path.join(pluginDirectory, f)) and not f.startswith(".") and not f.startswith("CVS")]
        failedToLoad = []
        for di in dirs:
            try:
                module = __import__(di, globals(), locals(), "Vispa.Plugins")
                self._loadablePlugins[module.plugin.__name__] = module.plugin
            except ImportError:
                logging.warning('Application: cannot load plugin ' + di + ': ' + exception_traceback())
                failedToLoad.append(di)
            except PluginIgnoredException,e:
                logging.info('Application: plugin ' + di + ' cannot be loaded and is ignored: ' + str(e))
            except AttributeError,e:
                logging.info('Application: plugin ' + di + ' is deactivated (define plugin in __init__.py to activate): ' + str(e))
                
        for pluginName in self._loadablePlugins.keys():
            # loop over all loadable plugins
            # this mechanism enables plugins to call initializePlugin() for plugins they depend on
            if not self.initializePlugin(pluginName):
                failedToLoad.append(pluginName)
        
        if len(failedToLoad) > 0:
            self.errorMessage(self.FAILED_LOADING_PLUGINS_ERROR + "\n".join(failedToLoad))
                
        self._collectFileExtensions()
        
    def initializePlugin(self, name):
        if name in [plugin.__class__.__name__ for plugin in self._plugins]:
            logging.info("%s: initalizePlugin(): Plugin '%s' already loaded. Aborting..." % (self.__class__.__name__, name))
            return True
        if not name in self._loadablePlugins.keys():
            logging.error("%s: initalizePlugin(): Unknown plugin '%s'. Aborting..." % (self.__class__.__name__, name))
            return False
        
        try:
            pluginObject = self._loadablePlugins[name](self)
            self._plugins.append(pluginObject)
            logging.debug('Application: added plugin ' + name)
            return True
        except ValueError:
            logging.warning('Application: ' + name + ' is not a plugin: ' + exception_traceback())
        return False
        
                    
    def plugins(self):
        return self._plugins
    
    def plugin(self, name):
        """ Returns plugin with given name or None if there is no such one.
        """
        if not name.endswith("Plugin"):
            name += "Plugin"
            
        for plugin in self._plugins:
            if name == plugin.__class__.__name__:
                return plugin
        return None

    def tabControllers(self):
        controllers=[self._window.tabWidget().widget(i).controller() for i in range(0, self._window.tabWidget().count())]
        controllers+=[tab.controller() for tab in self.mainWindow().tabWidgets()]
        return controllers
    
    def setCurrentTabController(self, controller):
        if controller.tab().tabWidget():
            self._window.activateWindow()
            self._window.tabWidget().setCurrentIndex(self.tabControllers().index(controller))
        else:
            controller.tab().activateWindow()
    
    def currentTabController(self):
        """ Return the TabController that belongs to the tab selected in the MainWindow.
        """
        #logging.debug('Application: currentTabController()')
        if isinstance(self.activeWindow(),AbstractTab):
            return self.activeWindow().controller()
        else:
            currentWidget = self._window.tabWidget().currentWidget()
            if isinstance(currentWidget, AbstractTab):
                return currentWidget.controller()
        raise NoCurrentTabControllerException
    
    def mainWindow(self):
        return self._window

    #def editPreferences(self):
    #    self.preferencesEditor = PreferencesEditor()
    #    self.preferencesEditor.show()

    def createAction(self, name, slot=None, shortcut=None, image=None, enabled=True):
        """ create an action with name and icon and connect it to a slot.
        """
        #logging.debug('Application: createAction() - ' + name)
        if image:
            image0 = QPixmap()
            image0.load(":/resources/" + image + ".svg")
            action = QAction(QIcon(image0), name, self._window)
        else:
            action = QAction(name, self._window)
        action.setEnabled(enabled)
        if slot:
            self.connect(action, SIGNAL("triggered()"), slot)
        if shortcut:
            if isinstance(shortcut, list):
                action.setShortcuts(shortcut)
            else:
                action.setShortcut(shortcut)
        return action
        
    def _fillFileMenu(self):
        """Called for the first time this function creates the file menu and fill it.

        The function is written in a way that it recreates the whole menu, if it
        is called again later during execution. So it is possible to aad new
        plugins and use them  (which means they appear in the menus) without
        restarting the program.
        """
        logging.debug('Application: _fillFileMenu()')
        self._fileMenuItems = {}
        if not self._window.fileMenu().isEmpty():
            self._window.fileMenu().clear()
        
        # New
        newFileActions = []
        for plugin in self._plugins:
            newFileActions += plugin.getNewFileActions()
        
        if len(newFileActions) == 1:
            newFileActions[0].setShortcut('Ctrl+N')
        
        self._window.fileMenu().addActions(newFileActions)               
        
        # Open
        openFileAction = self.createAction('&Open File', self.openFileDialog, 'Ctrl+O', "fileopen")
        self._window.fileMenu().addAction(openFileAction)
        self._window.fileToolBar().addAction(openFileAction)

        # Reload
        self._fileMenuItems['reloadFileAction'] = self.createAction('&Reload File', self.reloadFile, ['Ctrl+R', 'F5'], "reload")
        self._window.fileMenu().addAction(self._fileMenuItems['reloadFileAction'])
        #self._window.fileToolBar().addAction(self._fileMenuItems['reloadFileAction'])
        
        # Recent files
        if not hasattr(self, 'recentFilesMenu'):
            self._recentFilesMenu = QMenu('&Recent Files', self._window)
            self._recentFilesMenuActions = []
            for i in range(0, self.MAX_VISIBLE_RECENT_FILES):
                action = self.createAction("recent file " + str(i), self.openRecentFileSlot)
                action.setVisible(False)
                self._recentFilesMenu.addAction(action)                
                self._recentFilesMenuActions.append(action)
            self._recentFilesMenu.addSeparator()
            self._fileMenuItems['clearMissingRecentFilesAction'] = self.createAction("Clear missing files", self.clearMissingRecentFiles)
            self._recentFilesMenu.addAction(self._fileMenuItems['clearMissingRecentFilesAction'])
            self._fileMenuItems['clearRecentFilesAction'] = self.createAction("Clear list", self.clearRecentFiles)
            self._recentFilesMenu.addAction(self._fileMenuItems['clearRecentFilesAction'])
                
        self._window.fileMenu().addMenu(self._recentFilesMenu)

        self._window.fileMenu().addSeparator()
        
        # Close
        self._fileMenuItems['closeFileAction'] = self.createAction('&Close', self.closeFile, 'Ctrl+W', "closefile")
        self._window.fileMenu().addAction(self._fileMenuItems['closeFileAction'])
        
        # Close all
        self._fileMenuItems['closeAllAction'] = self.createAction('Close All', self.closeAllFiles, 'Ctrl+Shift+W', "closefileall")      
        self._window.fileMenu().addAction(self._fileMenuItems['closeAllAction'])
        
        self._window.fileMenu().addSeparator()
        
        # Save
        self._fileMenuItems['saveFileAction'] = self.createAction('&Save', self.saveFile, 'Ctrl+S', "filesave")      
        self._window.fileMenu().addAction(self._fileMenuItems['saveFileAction'])
        self._window.fileToolBar().addAction(self._fileMenuItems['saveFileAction'])
        
        # Save as
        self._fileMenuItems['saveFileAsAction'] = self.createAction('Save As...', self.saveFileAsDialog, 'Ctrl+Shift+S', image="filesaveas")      
        self._window.fileMenu().addAction(self._fileMenuItems['saveFileAsAction'])
   
        # Save all
        self._fileMenuItems['saveAllFilesAction'] = self.createAction('Save &All', self.saveAllFiles, "Ctrl+Alt+S", "filesaveall")      
        self._window.fileMenu().addAction(self._fileMenuItems['saveAllFilesAction'])
       
        self._window.fileMenu().addSeparator()
        
        #editPreferencesAction = self.createAction('Preferences',self.editPreferences)
        #self._window.fileMenu().addAction(editPreferencesAction)
        # Exit
        exit = self.createAction('&Exit', self.exit, "Ctrl+Q", "exit")      
        self._window.fileMenu().addAction(exit)

    def _fillEditMenu(self):
        """Called for the first time this function creates the edit menu and fills it.
        
        The function is written in a way that it recreates the whole menu, if it
        is called again later during execution. So it is possible to aad new
        plugins and use them  (which means they appear in the menus) without
        restarting the program.
        """
        logging.debug('Application: _fillEditMenu()')
        self._editMenuItems = {}
        if not self._window.editMenu().isEmpty():
            self._window.editMenu().clear()
            
        # Undo / Redo
        self._editMenuItems["undoAction"] = self.createAction("Undo", self.undoEvent, "Ctrl+Z", "edit-undo")
        self._editMenuItems["redoAction"] = self.createAction("Redo", self.redoEvent, "Ctrl+Y", "edit-redo")
        self._editMenuItems["undoAction"].setData(QVariant(1))
        self._editMenuItems["redoAction"].setData(QVariant(1))
        self._editMenuItems["undoAction"].setEnabled(False)
        self._editMenuItems["redoAction"].setEnabled(False)
        self._window.editMenu().addAction(self._editMenuItems["undoAction"])
        self._window.editMenu().addAction(self._editMenuItems["redoAction"])
        #self._editMenuItems["undoAction"].menu().addAction(self.createAction("test"))
        #self._editMenuItems["undoAction"].menu().setEnabled(False)
        #self._editMenuItems["undoAction"].menu().setVisible(False)
        
        self._undoActionsMenu = QMenu(self._window)
        self._undoMenuActions = []
        for i in range(0, self.MAX_VISIBLE_UNDO_EVENTS):
            action = self.createAction("undo " + str(i), self.undoEvent)
            action.setVisible(False)
            self._undoActionsMenu.addAction(action)                
            self._undoMenuActions.append(action)

        self._redoActionsMenu  = QMenu(self._window)
        self._redoMenuActions = []
        for i in range(0, self.MAX_VISIBLE_UNDO_EVENTS):
            action = self.createAction("redo " + str(i), self.redoEvent)
            action.setVisible(False)
            self._redoActionsMenu.addAction(action)                
            self._redoMenuActions.append(action)
                    
        # Cut
        self._editMenuItems['cutAction'] = self.createAction('&Cut', self.cutEvent, 'Ctrl+X', 'editcut')      
        self._window.editMenu().addAction(self._editMenuItems['cutAction'])
        self._editMenuItems['cutAction'].setEnabled(False)

        # Copy
        self._editMenuItems['copyAction'] = self.createAction('C&opy', self.copyEvent, 'Ctrl+C', 'editcopy')      
        self._window.editMenu().addAction(self._editMenuItems['copyAction'])
        self._editMenuItems['copyAction'].setEnabled(False)
        
        # Paste
        self._editMenuItems['pasteAction'] = self.createAction('&Paste', self.pasteEvent, 'Ctrl+V', 'editpaste')      
        self._window.editMenu().addAction(self._editMenuItems['pasteAction'])
        self._editMenuItems['pasteAction'].setEnabled(False)
        
        # Select all
        self._editMenuItems['selectAllAction'] = self.createAction("Select &all", self.selectAllEvent, "Ctrl+A", "selectall")
        self._window.editMenu().addAction(self._editMenuItems['selectAllAction'])
        self._editMenuItems['selectAllAction'].setVisible(False)

        self._window.editMenu().addSeparator()
        
        # Find
        self._editMenuItems['findAction'] = self.createAction('&Find', self.findEvent, 'Ctrl+F', "edit-find")      
        self._window.editMenu().addAction(self._editMenuItems['findAction'])
        self._editMenuItems['findAction'].setEnabled(False)

    def _fillHelpMenu(self):
        logging.debug('Application: _fillHelpMenu()')
        self._helpMenuItems = {}
                    
        # About
        self._helpMenuItems['aboutAction'] = self.createAction('&About', self.aboutBoxSlot, 'F1')      
        self._window.helpMenu().addAction(self._helpMenuItems['aboutAction'])
        
        # open log file
        if self._logFile:
            self._helpMenuItems['openLogFile'] = self.createAction("Open log file", self.openLogFileSlot)      
            self._window.helpMenu().addAction(self._helpMenuItems['openLogFile'])
        
        # Offline Documentation
        if os.path.exists(os.path.join(docDirectory,"index.html")):
            self._window.helpMenu().addAction(self.createAction('Offline Documentation', self._openDocumentation, "CTRL+F1"))
        
        # Vispa Website
        self._window.helpMenu().addAction(self.createAction('Website', self._openWebsite, "Shift+F1"))
        
    def updateMenu(self):
        """ Update recent files and enable disable menu entries in file and edit menu.
        """
        #logging.debug('Application: updateMenu()')
        if self.mainWindow().startupScreen():
            self.updateStartupScreen()
        # Recent files
        num_recent_files = min(len(self._recentFiles), self.MAX_VISIBLE_RECENT_FILES)
        for i in range(0, num_recent_files):
            filename = self._recentFiles[i]
            self._recentFilesMenuActions[i].setText(os.path.basename(filename))
            self._recentFilesMenuActions[i].setToolTip(filename)
            self._recentFilesMenuActions[i].setStatusTip(filename)
            self._recentFilesMenuActions[i].setData(QVariant(filename))
            self._recentFilesMenuActions[i].setVisible(True)
            
        for i in range(num_recent_files, self.MAX_VISIBLE_RECENT_FILES):
            self._recentFilesMenuActions[i].setVisible(False)
            
        if num_recent_files == 0:
            self._fileMenuItems['clearRecentFilesAction'].setEnabled(False)
            self._fileMenuItems['clearMissingRecentFilesAction'].setEnabled(False)
        else:
            self._fileMenuItems['clearRecentFilesAction'].setEnabled(True)
            self._fileMenuItems['clearMissingRecentFilesAction'].setEnabled(True)
            
        # Enabled / disable menu entries depending on number of open files
        at_least_one_flag = False
        at_least_two_flag = False
        if len(self.tabControllers()) > 1:
            at_least_one_flag = True
            at_least_two_flag = True
        elif len(self.tabControllers()) > 0:
            at_least_one_flag = True
        
        self._fileMenuItems['saveFileAction'].setEnabled(at_least_one_flag)
        self._fileMenuItems['saveFileAsAction'].setEnabled(at_least_one_flag)
        self._fileMenuItems['reloadFileAction'].setEnabled(at_least_one_flag)
        self._fileMenuItems['closeFileAction'].setEnabled(at_least_one_flag)
        
        self._fileMenuItems['saveAllFilesAction'].setEnabled(at_least_two_flag)
        self._fileMenuItems['closeAllAction'].setEnabled(at_least_two_flag)
        
        try:
            if at_least_one_flag:
                if not self.currentTabController().isEditable():
                    self._fileMenuItems['saveFileAction'].setEnabled(False)
                    self._fileMenuItems['saveFileAsAction'].setEnabled(False)
                if not self.currentTabController().isModified():
                    self._fileMenuItems['saveFileAction'].setEnabled(False)
                
            # Copy / Cut / Paste
            copy_paste_enabled_flag = at_least_one_flag and self.currentTabController().isCopyPasteEnabled()
            self._editMenuItems['cutAction'].setEnabled(copy_paste_enabled_flag)
            self._editMenuItems['copyAction'].setEnabled(copy_paste_enabled_flag)
            self._editMenuItems['pasteAction'].setEnabled(copy_paste_enabled_flag)
            
            self._editMenuItems['selectAllAction'].setVisible(self.currentTabController().allowSelectAll())
            
            self._editMenuItems['findAction'].setEnabled(at_least_one_flag and self.currentTabController().isFindEnabled())
            
            # Undo / Redo
            undo_supported_flag = at_least_one_flag and self.currentTabController().supportsUndo()
            self._editMenuItems["undoAction"].setEnabled(undo_supported_flag)
            self._editMenuItems["undoAction"].setVisible(undo_supported_flag)
            self._editMenuItems["redoAction"].setEnabled(undo_supported_flag)
            self._editMenuItems["redoAction"].setVisible(undo_supported_flag)
            self.showPluginToolBar(self._undoToolBar, undo_supported_flag)
            
            if undo_supported_flag:
                undo_events = self.currentTabController().undoEvents()
                num_undo_events = min(len(undo_events), self.MAX_VISIBLE_UNDO_EVENTS)
                self._editMenuItems["undoAction"].setEnabled(num_undo_events > 0)
                if num_undo_events > 1:
                    self._editMenuItems["undoAction"].setMenu(self._undoActionsMenu)
                else:
                    self._editMenuItems["undoAction"].setMenu(None)
                for i in range(0, num_undo_events):
                    undo_event = undo_events[num_undo_events - i - 1]   # iterate backwards
                    self._undoMenuActions[i].setText(undo_event.LABEL)
                    self._undoMenuActions[i].setToolTip(undo_event.description())
                    self._undoMenuActions[i].setStatusTip(undo_event.description())
                    self._undoMenuActions[i].setData(QVariant(i+1))
                    self._undoMenuActions[i].setVisible(True)
                for i in range(num_undo_events, self.MAX_VISIBLE_UNDO_EVENTS):
                    self._undoMenuActions[i].setVisible(False)
                
                redo_events = self.currentTabController().redoEvents()
                num_redo_events = min(len(redo_events), self.MAX_VISIBLE_UNDO_EVENTS)
                self._editMenuItems["redoAction"].setEnabled(num_redo_events > 0)
                if num_redo_events > 1:
                    self._editMenuItems["redoAction"].setMenu(self._redoActionsMenu)
                else:
                    self._editMenuItems["redoAction"].setMenu(None)
                for i in range(0, num_redo_events):
                    redo_event = redo_events[num_redo_events - i - 1]   # iterate backwards
                    self._redoMenuActions[i].setText(redo_event.LABEL)
                    self._redoMenuActions[i].setToolTip(redo_event.description())
                    self._redoMenuActions[i].setStatusTip(redo_event.description())
                    self._redoMenuActions[i].setData(QVariant(i+1))
                    self._redoMenuActions[i].setVisible(True)
                for i in range(num_redo_events, self.MAX_VISIBLE_UNDO_EVENTS):
                    self._redoMenuActions[i].setVisible(False)
            
        except NoCurrentTabControllerException:
            pass
            
    def _openDocumentation(self):
        """ Opens Vispa Offline Documentation
        """
        webbrowser.open(os.path.join(docDirectory,"index.html"), 2, True)
        
    def _openWebsite(self):
        """ Open new browser tab and opens Vispa Project Website.
        """
        webbrowser.open(websiteUrl, 2, True)
        
    def createPluginMenu(self, name):
        """ Creates menu in main window's menu bar before help menu and adds it to _pluginMenus list.
        """
        menu = QMenu(name)
        self._window.menuBar().insertMenu(self._window.helpMenu().menuAction(), menu)
        self._pluginMenus.append(menu)
        return menu
    
    def showPluginMenu(self, menuObject, show=True):
        """ Shows given menu if it is in _pluginMenus list.
        """
        #logging.debug(self.__class__.__name__ +": showPluginMenu()")
        if menuObject in self._pluginMenus:
            # show all actions and activate their shortcuts
            if show:
                for action in menuObject.actions():
                    if hasattr(action,"_wasVisible") and action._wasVisible!=None:
                        action.setVisible(action._wasVisible)
                        action._wasVisible=None
                    else:
                        action.setVisible(True)     # has to be after actions() loop to prevent permanant invisibility on Mac OS X
            menuObject.menuAction().setVisible(show)
        
    def hidePluginMenu(self, menuObject):
        """ Hides given menu object if it it in _pluginMenus list.
        """
        self.showPluginMenu(menuObject, False)
            
    def hidePluginMenus(self):
        """ Hides all menus in _pluginMenus list.
        """
        for menuObject in self._pluginMenus:
            # hide all actions and deactivate their shortcuts
            menuObject.menuAction().setVisible(False)
            for action in menuObject.actions():
                if not hasattr(action,"_wasVisible") or action._wasVisible==None:
                    action._wasVisible=action.isVisible()
                action.setVisible(False)    # setVisible() here hides plugin menu forever on Mac OS X (10.5.7), Qt 4.5.
    
    def createPluginToolBar(self, name):
        """ Creates tool bar in main window and adds it to _pluginToolBars list.
        """
        toolBar = self._window.addToolBar(name)
        self._pluginToolBars.append(toolBar)
        return toolBar

    def showPluginToolBar(self, toolBarObject, show=True):
        """ Shows given toolbar if it is in _pluginToolBars list.
        """
        if toolBarObject in self._pluginToolBars:
            toolBarObject.setVisible(show)
            
    def hidePluginMenu(self, toolBarObject):
        """ Hides given toolbar object if it it in _pluginToolBars list.
        """
        self.showPluginToolBar(toolBarObject, False)
        #if toolBarObject in self._pluginToolBars:
        #    toolBarObject.menuAction().setVisible(False)
            
    def hidePluginToolBars(self):
        """ Hides all toolbars in _toolBarMenus list.
        """
        for toolBar in self._pluginToolBars:
            toolBar.hide()
            
    def createZoomToolBar(self):
        """ Creates tool bar with three buttons "user", "100 %" and "all".
        
        See TabController's documentation of zoomUser(), zoomHundred() and zoomAll() to find out more on the different zoom levels.
        """
        self._zoomToolBar = self.createPluginToolBar('Zoom ToolBar')
        self._zoomToolBar.addAction(self.createAction('Revert Zoom', self.zoomUserEvent, image='zoomuser'))
        self._zoomToolBar.addAction(self.createAction('Zoom to 100 %', self.zoomHundredEvent, image='zoom100'))
        self._zoomToolBar.addAction(self.createAction('Zoom to all', self.zoomAllEvent, image='zoomall'))
    
    def showZoomToolBar(self):
        """ Makes zoom tool bar visible.
        
        Should be called from TabController's selected() function, if the controller wants to use the tool bar.
        """
        self.showPluginToolBar(self._zoomToolBar)
    
    def hideZoomToolBar(self):
        """ Makes zoom tool bar invisible.
        """
        self._zoomToolBar.hide()
        
    def createUndoToolBar(self):
        """ Creates tool bar with buttons to invoke undo and redo events.
        
        Needs to be called after _fillEditMenu() as actions are defined there.
        """
        self._undoToolBar = self.createPluginToolBar("Undo ToolBar")
        self._undoToolBar.addAction(self._editMenuItems["undoAction"])
        self._undoToolBar.addAction(self._editMenuItems["redoAction"])
        
    def showUndoToolBar(self):
        """ Makes undo tool bar visible.
        """
        self.showPluginToolBar(self._undoToolBar)
        
    def hideUndoToolBar(self):
        """ Hides undo tool bar.
        """
        self._undoToolBar.hide()
    
    def clearRecentFiles(self):
        """ Empties list of recent files and updates main menu.
        """ 
        self._recentFiles = []
        self._saveIni()
        self.updateMenu()
        
    def clearMissingRecentFiles(self):
        """ Removes entries from recent files menu if file does no longer exist.
        """
        newList = []
        for file in self._recentFiles:
            if os.path.exists(file):
                newList.append(file)
        self._recentFiles = newList
        self._saveIni()
        self.updateMenu()

    def addRecentFile(self, filename):
        """ Adds given filename to list of recent files.
        """
        logging.debug('Application: addRecentFile() - ' + filename)
        if isinstance(filename, QString):
            filename = str(filename)    # Make sure filename is a python string not a QString
        leftCount = self.MAX_RECENT_FILES - 1
        if filename in self._recentFiles:
            del self._recentFiles[self._recentFiles.index(filename)]
        self._recentFiles = [filename] + self._recentFiles[:leftCount]
        self._saveIni()
       
    def recentFiles(self):
        """ Returns list of recently opened files.
        """
        return self._recentFiles    
    
    def getLastOpenLocation(self):
        """ Returns directory name of first entry of recent files list.
        """
        # if current working dir is vispa directory use recentfile or home
        if os.path.abspath(os.getcwd()) in [os.path.abspath(baseDirectory),os.path.abspath(os.path.join(baseDirectory,"bin"))] or platform.system() == "Darwin":
            if len(self._recentFiles) > 0:
                return os.path.dirname(self._recentFiles[0])
            elif platform.system() == "Darwin":
                # Mac OS X
                return homeDirectory + "/Documents"
            else:
                return homeDirectory
        # if user navigated to another directory use this
        else:
            return os.getcwd()
    
    def recentFilesFromPlugin(self,plugin):
        files=[]
        filetypes = plugin.filetypes()
        extension=None
        if len(filetypes) > 0:
            extension=filetypes[0].extension().lower()
        for file in self._recentFiles:
            if os.path.splitext(os.path.basename(file))[1][1:].lower()==extension:
                files+=[file]
        return files
    
    def updateStartupScreen(self):
        screen=self.mainWindow().startupScreen()
        screen.analysisDesignerRecentFilesList().clear()
        screen.analysisDesignerRecentFilesList().addItem("...")
        screen.analysisDesignerRecentFilesList().setCurrentRow(0)
        plugin=self.plugin("AnalysisDesignerPlugin")
        if plugin:
            files = self.recentFilesFromPlugin(plugin)
            for file in files:
                screen.analysisDesignerRecentFilesList().addItem(os.path.basename(file))
        
        screen.pxlEditorRecentFilesList().clear()
        screen.pxlEditorRecentFilesList().addItem("...")
        screen.pxlEditorRecentFilesList().setCurrentRow(0)
        plugin=self.plugin("PxlPlugin")
        if plugin:
            files = self.recentFilesFromPlugin(plugin)
            for file in files:
                screen.pxlEditorRecentFilesList().addItem(os.path.basename(file))
        
    def exit(self):
        self._window.close()

    def shutdownPlugins(self):
        logging.debug('Application: shutting down plugins' )
        for plugin in self._plugins:
          plugin.shutdown()


    def _collectFileExtensions(self):
        """ Loop over all plugins and remember their file extensions.
        """
        self._knownExtensionsDictionary = {}
        self._knownFiltersList = []
        self._knownFiltersList.append('All files (*.*)')
        for plugin in self.plugins():
            for ft in plugin.filetypes():
                self._knownExtensionsDictionary[ft.extension()] = plugin
                self._knownFiltersList.append(ft.fileDialogFilter())
        if len(self._knownFiltersList) > 0:
            allKnownFilter = 'All known files (*.' + " *.".join(self._knownExtensionsDictionary.keys()) + ')'
            self._knownFiltersList.insert(1, allKnownFilter)
            logging.debug('Application: _collectFileExtensions() - ' + allKnownFilter)
        else:
            logging.debug('Application: _collectFileExtensions()')
        
        
    def openFileDialog(self, defaultFileFilter=None):
        """Displays a common open dialog for all known file types.
        """
        logging.debug('Application: openFileDialog()')
        
        if not defaultFileFilter:
            if len(self._knownFiltersList) > 1:
                # Set defaultFileFilter to all known files
                defaultFileFilter = self._knownFiltersList[1]
            else:
                # Set dfaultFileFilter to any file type
                defaultFileFilter = self._knownFiltersList[0]
        
        # Dialog
        filename = QFileDialog.getOpenFileName(
                                               self._window,
                                               'Select a file',
                                               self.getLastOpenLocation(),
                                               ";;".join(self._knownFiltersList),
                                               defaultFileFilter)
        if not filename.isEmpty():
            self.openFile(filename)

    def openFile(self, filename):
        """ Decides which plugin should handle opening of the given file name.
        """
        logging.debug('Application: openFile()')
        statusMessage = self.startWorking("Opening file " + filename)
        if isinstance(filename, QString):
            filename = str(filename)  # convert QString to Python String
        
        # Check whether file is already opened
        for controller in self.tabControllers():
            if filename == controller.filename():
                self.setCurrentTabController(controller)
                self.stopWorking(statusMessage, "already open")
                return
        
        baseName = os.path.basename(filename)
        ext = os.path.splitext(baseName)[1].lower().strip(".")
        errormsg = None

        if self._knownExtensionsDictionary == {}:
            self._collectFileExtensions()
        
        foundCorrectPlugin = False
        if os.path.exists(filename):
            if ext in self._knownExtensionsDictionary:
                foundCorrectPlugin = True
                try:
                    if self._knownExtensionsDictionary[ext].openFile(filename):
                        self.addRecentFile(filename)
                    else:
                        logging.error(self.__class__.__name__ + ": openFile() - Error while opening '" + str(filename) + "'.")
                        self.errorMessage("Failed to open file.")
                except Exception:
                    logging.error(self.__class__.__name__ + ": openFile() - Error while opening '" + str(filename) + "' : " + exception_traceback())
                    self.errorMessage("Exception while opening file. See log for details.")

            if not foundCorrectPlugin:
                errormsg = 'Unknown file type (.' + ext + '). Aborting.'
        else:
            errormsg = 'File does not exist: ' + filename

        self.updateMenu()
        
        # Error messages
        if not errormsg:
            self.stopWorking(statusMessage)
        else:
            logging.error(errormsg)
            self.stopWorking(statusMessage, "failed")
            self.warningMessage(errormsg)
    
    def reloadFile(self):
        """ Tells current tab controller to refresh.
        """
        logging.debug('Application: reloadFile()')
        try:
            if self.currentTabController().filename() and self.currentTabController().allowClose():
                self.currentTabController().refresh()
                self.currentTabController().setModified(False)
        except NoCurrentTabControllerException:
            pass
        # call tabChanged instead of updateMenu to be qt 4.3 compatible
        self.tabChanged()
    
    def closeFile(self):
        """ Tells current tab controller to close.
        """
        logging.debug('Application: closeCurrentFile()')
        try:
            self.currentTabController().close()
        except NoCurrentTabControllerException:
            pass
        # call tabChanged instead of updateMenu to be qt 4.3 compatible
        self.tabChanged()
        
    def closeAllFiles(self):
        """ Closes all open tabs unless user aborts closing.
        """
        logging.debug('Application: closeAllFiles()')
        # to prevent unneeded updates set flag
        self._closeAllFlag = True
        while len(self.tabControllers())>0:
            controller=self.tabControllers()[0]
            if controller.close() == False:
                break
        self._closeAllFlag = False

        # call tabChanged instead of updateMenu to be qt 4.3 compatible
        self.tabChanged()
        
    def saveFile(self):
        """ Tells current tab controller to save its file.
        """
        logging.debug('Application: saveFile()')
        try:
            self.currentTabController().save()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
        
    def saveFileAsDialog(self):
        """This functions asks the user for a file name. 
        """
        logging.debug('Application: saveFileAsDialog()')
        try:
            currentTabController = self.currentTabController()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
            return

        if currentTabController.filename():
            startDirectory = currentTabController.filename()
        else:
            startDirectory = self.getLastOpenLocation()
                
        filetypesList = []
        for filetype in currentTabController.supportedFileTypes():
            filetypesList.append(Filetype(filetype[0], filetype[1]).fileDialogFilter())
        filetypesList.append('Any (*.*)')

        selectedFilter = QString("")
        filename = str(QFileDialog.getSaveFileName(
                                            self._window,
                                            'Select a file',
                                            startDirectory,
                                            ";;".join(filetypesList), selectedFilter))
        if filename != "":
            # add extension if necessary
            if os.path.splitext(filename)[1].strip(".") == "" and str(selectedFilter) != 'Any (*.*)':
                ext = currentTabController.supportedFileTypes()[filetypesList.index(str(selectedFilter))][0]
                filename = os.path.splitext(filename)[0] + "." + ext
            return currentTabController.save(filename)
        return False
        
    def saveAllFiles(self):
        """ Tells tab controllers of all tabs to save.
        """
        logging.debug('Application: saveAllFiles()')
        
        for controller in self.tabControllers():
            if controller.filename() or controller == self.currentTabController():
                controller.save()

    def cutEvent(self):
        """ Called when cut action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        try:
            self.currentTabController().cut()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
        
    def copyEvent(self):
        """ Called when copy action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        try:
            self.currentTabController().copy()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
        
    def pasteEvent(self):
        """ Called when paste action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        try:
            self.currentTabController().paste()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
            
    def selectAllEvent(self):
        """ Called when selectAll action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        try:
            self.currentTabController().selectAll()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
            
    def findEvent(self):
        """ Called when find action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        try:
            self.currentTabController().find()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
       
    def zoomUserEvent(self):
        """ Handles button pressed event from zoom tool bar and forwards it to current tab controller.
        """
        try:
            self.currentTabController().zoomUser()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
        
    def zoomHundredEvent(self):
        """ Handles button pressed event from zoom tool bar and forwards it to current tab controller.
        """
        try:
            self.currentTabController().zoomHundred()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
    
    def zoomAllEvent(self):
        """ Handles button pressed event from zoom tool bar and forwards it to current tab controller.
        """
        try:
            self.currentTabController().zoomAll()
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
         
    def undoEvent(self):
        """ Handles undo action for buttons in undo tool bar and edit menu.
        """
        try:
            num = 1
            sender = self.sender()
            if sender:
                num = sender.data().toInt()
                if len(num) > 1:
                    # strange: toInt returns tuple like (1, True), QT 4.6.0, Mac OS X 10.6.4, 2010-06-28
                    num = num[0]
            self.currentTabController().undo(num)
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": "+ self.TAB_PREMATURELY_CLOSED_WARNING) 
         
    def redoEvent(self):
        """ Handles redo action for buttons in undo tool bar and edit menu.
        """
        try:
            num = 1
            sender = self.sender()
            if sender:
                num = sender.data().toInt()
                if len(num) > 1:
                    # strange: toInt returns tuple like (1, True), QT 4.6.0, Mac OS X 10.6.4, 2010-06-28
                    num = num[0]
            self.currentTabController().redo(num)
        except NoCurrentTabControllerException:
            logging.warning(self.__class__.__name__ + ": "+ self.TAB_PREMATURELY_CLOSED_WARNING) 
         
    def aboutBoxSlot(self):
        """ Displays about box. 
        """
        logging.debug('Application: aboutBoxSlot()')
        about = AboutDialog(self)
        about.onScreen()
        
    def openLogFileSlot(self):
        if self._logFile:
            self.doubleClickOnFile(self._logFile)
        else:
            logging.warning("%s: openLogFileSlot(): _logFile not set. Aborting..." % self.__class__.__name__)

    def openRecentFileSlot(self):
        """ Slot for opening recent file.
        
        Called from recent file menu action. Filename is set as data object (QVariant) of action.
        """
        filename = self.sender().data().toString()
        logging.debug('Application: openRecentFileSlot() - ' + filename)
        self.openFile(filename)

    def tabChanged(self, tab=None):
        """ when a different tab is activated update menu
        """
        logging.debug('Application: tabChanged()')
        # only update once when closing all files at once
        if not self._closeAllFlag:
            self.hidePluginMenus()
            self.hidePluginToolBars()
            self.updateWindowTitle()
            self.updateMenu()
            try:
                self.currentTabController().activated()
                self.currentTabController().checkModificationTimestamp()
            except NoCurrentTabControllerException:
                pass
            
        self.mainWindow().setStartupScreenVisible(self.mainWindow().tabWidget().count() == 0)
    
    def updateMenuAndWindowTitle(self):
        """ Update menu and window title.
        """
        self.updateMenu()
        self.updateWindowTitle()

    def windowTitle(self):
        return str(self._window.windowTitle()).split("-")[0].strip()
    
    def updateWindowTitle(self):
        """ update window caption
        """
        #logging.debug('Application: updateWindowTitle()')
        name = self.windowTitle()
        
        try:
            filename = self.currentTabController().filename()
        except NoCurrentTabControllerException:
            filename = None
            
        if filename:
            dirName = os.path.dirname(sys.argv[0])
            if os.path.abspath(dirName) in filename:
                filename = filename[len(os.path.abspath(dirName)) + 1:]
            name = name + " - " + filename
        self._window.setWindowTitle(name)

    def ini(self):
        if not self._ini:
            self._ini = ConfigParser.ConfigParser()
            self._ini.read(self._iniFileName)
        return self._ini 

    def writeIni(self):
        try:
            configfile = open(self._iniFileName, "w")
            self._ini.write(configfile)
            configfile.close()
        except IOError:
            pass
        self._ini = None
        
    def _loadIni(self):
        """ Save the list of recent files.
        """
        logging.debug('Application: _loadIni()')
        ini = self.ini()
        self._recentFiles = []
        if ini.has_section("history"):
            for i in range(0, self.MAX_RECENT_FILES):
                if ini.has_option("history", str(i)):
                    self._recentFiles+=[ini.get("history", str(i))]
               
    def _saveIni(self):
        """ Load the list of recent files.
        """
        logging.debug('Application: _saveIni()')
        ini = self.ini()
        if ini.has_section("history"):
            ini.remove_section("history")
        ini.add_section("history")
        for i in range(len(self._recentFiles)):
            ini.set("history", str(i), self._recentFiles[i])
                
        self.writeIni()

    def errorMessage(self, message):
        """ Displays error message.
        """
        QMessageBox.critical(self.mainWindow(), 'Error', message)
        
    def warningMessage(self, message):
        """ Displays warning message.
        """
        QMessageBox.warning(self.mainWindow(), 'Warning', message)
        
    def infoMessage(self, message):
        """ Displays info message.
        """
        QMessageBox.about(self.mainWindow(), 'Info', message)
        
    def showMessageBox(self, text, informativeText="", standardButtons=QMessageBox.Ok | QMessageBox.Cancel | QMessageBox.Ignore, defaultButton=QMessageBox.Ok, extraButtons=None):
        """ Shows a standardized message box and returns the pressed button.
        
        See documentation on Qt's QMessageBox for a list of possible standard buttons.
        """
        
        msgBox = QMessageBox(self.mainWindow())
        msgBox.setParent(self.mainWindow(), Qt.Sheet)     # Qt.Sheet: Indicates that the widget is a Macintosh sheet.
        msgBox.setText(text)
        msgBox.setInformativeText(informativeText)
        msgBox.setStandardButtons(standardButtons)
        if extraButtons!=None:
            for button,role in extraButtons:
                msgBox.addButton(button,role)
        msgBox.setDefaultButton(defaultButton)
        return msgBox.exec_()
        
    def doubleClickOnFile(self, filename):
        """ Opens file given as argument if possible in Vispa.
        
        If Vispa cannot handle the file type the file will be opened in it's default application.
        """
        logging.debug(self.__class__.__name__ + ": doubleClickOnFile() - " + str(filename))
        
        if filename == "":
            return
        
        baseName = os.path.basename(filename)
        ext = os.path.splitext(baseName)[1].lower().strip(".")
        if self._knownExtensionsDictionary == {}:
            self._collectFileExtensions()
        if os.path.exists(filename):
            if ext in self._knownExtensionsDictionary:
                return self.openFile(filename)
        
        # open file in default application
        try:
          if 'Windows' in platform.system():
              os.startfile(filename)
          elif 'Darwin' in platform.system():
            if os.access(filename, os.X_OK):
              logging.warning("It seems that executing the python program is the default action on this system, which is processed when double clicking a file. Please change that to open the file witrh your favourite editor, to use this feature.")
            else:
              subprocess.call(("open", filename))
          elif 'Linux' in platform.system():
          # Linux comes with many Desktop Enviroments
            if os.access(filename, os.X_OK):
              logging.warning("It seems that executing the python program is the default action on this system, which is processed when double clicking a file. Please change that to open the file witrh your favourite editor, to use this feature.")
            else:
              try:
                  #Freedesktop Standard
                  subprocess.call(("xdg-open", filename))
              except:
                try:
                   subprocess.call(("gnome-open", filename))
                except:
                   logging.error(self.__class__.__name__ + ": doubleClickOnFile() - Platform '" + platform.platform() + "'. Cannot open file. I Don't know how!")
        except:
          logging.error(self.__class__.__name__ + ": doubleClickOnFile() - Platform '" + platform.platform() + "'. Error while opening file: " + str(filename))

    def createStatusBar(self):
        self._workingMessages = {}
        
        self._progressWidget = RotatingIcon(":/resources/vispabutton.png")
        self._window.statusBar().addPermanentWidget(self._progressWidget)

    def startWorking(self, message=""):
        if len(self._workingMessages.keys()) == 0:
            self._progressWidget.start()
        self._window.statusBar().showMessage(message + "...")
        self._messageId+=1
        self._workingMessages[self._messageId] = message
        self._progressWidget.setToolTip(message)
        return self._messageId

    def stopWorking(self, id, end="done"):
        if not id in self._workingMessages.keys():
            logging.error(self.__class__.__name__ +": stopWorking() - Unknown id %s. Aborting..." % str(id))
            return
        if len(self._workingMessages.keys()) > 1:
            self._window.statusBar().showMessage(self._workingMessages[self._workingMessages.keys()[0]] + "...")
            self._progressWidget.setToolTip(self._workingMessages[self._workingMessages.keys()[0]])
        else:
            self._progressWidget.stop()
            self._progressWidget.setToolTip("")
            self._window.statusBar().showMessage(self._workingMessages[id] + "... " + end + ".")
        del self._workingMessages[id]
            
    def tabCloseRequest(self, i):
        self.mainWindow().tabWidget().setCurrentIndex(i)
        self.closeFile()

    def showStatusMessage(self, message, timeout=0):
        self._window.statusBar().showMessage(message, timeout)

    def cancel(self):
        """ Cancel operations in current tab.
        """
        logging.debug(__name__ + ": cancel")
        try:
            self.currentTabController().cancel()
        except NoCurrentTabControllerException:
            pass
