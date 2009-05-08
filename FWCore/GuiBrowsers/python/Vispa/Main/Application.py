import os
import sys
import logging
import logging.handlers
import ConfigParser
import optparse
import webbrowser
import subprocess

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Directories import *
from MainWindow import *
from AbstractTab import *
from Filetype import *
from Exceptions import *

import Resources

class Application(QApplication):

    MAX_RECENT_FILES = 10
    FAILED_LOADING_PLUGINS_ERROR = "Errors while loading plugins. For details see error output or log file.\n\nThe following plugins won't work correctly:\n\n"
    TAB_PREMATURELY_CLOSED_WARNING = "Tab was closed before user request could be handled."

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
        self._commandLineParser = optparse.OptionParser()
        self._commandLineOptions = None

        self._initLogging()
        self._loadIni()

        self._window = MainWindow(self, applicationName)
        self._connectSignals()
        
        self._loadPlugins()
        
        self._fillFileMenu()
        self._fillEditMenu()
        self._fillHelpMenu()
        
        self.updateMenu()
        self.createZoomToolBar()
        self.hidePluginMenus()
        self.hidePluginToolBars()
        self.createStatusBar()
    
        self._readCommandLineAttributes()

        logging.debug('Running with Qt-Version ' + str(qVersion()))
        
    def commandLineParser(self):
        return self._commandLineParser
        
    def commandLineOptions(self):
        return self._commandLineOptions
        
    def setVersion(self, version):
        self._version = version

    def _readCommandLineAttributes(self):
        """ Analyzes the command line attributes and print usage summary if required.
        """
        self._commandLineParser.add_option("-f", "--file", dest="filename", help="open a FILE", metavar="FILE")
        self._commandLineParser.add_option("-l", "--loglevel", dest="loglevel", help="set LOGLEVEL to 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL", metavar="LOGLEVEL", type="int")
        (self._commandLineOptions, args) = self._commandLineParser.parse_args()
        if self._commandLineOptions.filename:
            self.openFile(self._commandLineOptions.filename)
        if self._commandLineOptions.loglevel:
            logging.root.setLevel(self._commandLineOptions.loglevel)
        
        if len(args) > 0:
            self.openFile(args[0])
        
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
        
        handler1 = logging.handlers.RotatingFileHandler(logfile, maxBytes=100000, backupCount=1)
        formatter1 = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler1.setFormatter(formatter1)

        handler2 = logging.StreamHandler(sys.stderr)
        formatter2 = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler2.setFormatter(formatter2)
        
        logging.root.handlers = []
        logging.root.addHandler(handler1)
        logging.root.addHandler(handler2)

        self._infologger = logging.getLogger("info")
        self._infologger.setLevel(logging.INFO)
        self._infologger.handlers = []
        self._infologger.info("Start logging to " + logfile)
        
    def run(self):
        """ Show the MainWindow and run the application.
        """
        logging.debug('Application: run()')

        self._window.show()
        self.setActiveWindow(self._window)
        self.exec_()
        self._infologger.info("Stop logging")

    def _connectSignals(self):
        """ Connect signal to observe the TabWidget in the MainWindow.
        """
        logging.debug('Application: _connectSignals()')
        self.connect(self._window.tabWidget(), SIGNAL("currentChanged(int)"), self.tabChanged)
        self.connect(self._window, SIGNAL("activated()"), self.tabChanged)
        
    def _loadPlugins(self):
        """ Search all subfolders of the plugin directory for vispa plugins and registers them.
        """
        logging.debug('Application: _loadPlugins()')
        dirs = ["Vispa.Plugins." + str(f) for f in os.listdir(pluginDirectory)
                if os.path.isdir(os.path.join(pluginDirectory, f)) and not f.startswith(".")]
        failedToLoad = []
        for di in dirs:
            try:
                module = __import__(di, globals(), locals(), "Vispa.Plugins")
                pluginObject = module.plugin(self)
                self._plugins.append(pluginObject)
                logging.debug('Application: added plugin ' + di)
            except AttributeError:
                logging.info('Application: plugin ' + di + ' is deactivated (define plugin in __init__.py to activate): ' + exception_traceback())
            except PluginIgnoredException:
                logging.info('Application: plugin ' + di + ' cannot be loaded and is ignored: ' + exception_traceback())
            except ValueError:
                logging.warning('Application: ' + di + ' is not a plugin: ' + exception_traceback())
                failedToLoad.append(di)
            except ImportError:
                logging.warning('Application: cannot load plugin ' + di + ': ' + exception_traceback())
                failedToLoad.append(di)
        
        if len(failedToLoad) > 0:
            self.errorMessage(self.FAILED_LOADING_PLUGINS_ERROR + "\n".join(failedToLoad))
                
        self._collectFileExtensions()
                    
    def plugins(self):
        return self._plugins
    
    def currentTabController(self):
        """ Return the TabController that belongs to the tab selected in the MainWindow.
        """
        
        logging.debug('Application: currentTabController()')
        currentWidget = self._window.tabWidget().currentWidget()
        if isinstance(currentWidget, AbstractTab):
            return currentWidget.controller()
        raise NoCurrentTabControllerException
    
    def mainWindow(self):
        return self._window

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
            for i in range(0, self.MAX_RECENT_FILES):
                action = self.createAction("recent file " + str(i), self.openRecentFile)
                action.setVisible(False)
                self._recentFilesMenu.addAction(action)                
                self._recentFilesMenuActions.append(action)
            self._recentFilesMenu.addSeparator()
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

        self._window.editMenu().addSeparator()
        
        # Find
        self._editMenuItems['findAction'] = self.createAction('&Find', self.findEvent, 'Ctrl+F', "edit-find")      
        self._window.editMenu().addAction(self._editMenuItems['findAction'])
        self._editMenuItems['findAction'].setEnabled(False)

    def _fillHelpMenu(self):
        logging.debug('Application: _fillHelpMenu()')
        self._helpMenuItems = {}
                    
        # About
        self._helpMenuItems['aboutAction'] = self.createAction('&About', self.aboutBox, 'F1')      
        self._window.helpMenu().addAction(self._helpMenuItems['aboutAction'])
        
        # Vispa Website
        self._window.helpMenu().addAction(self.createAction('Website', self._openWebsite, "Shift+F1"))
        
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
    
    def showPluginMenu(self, menuObject):
        """ Shows given menu if it is in _pluginMenus list.
        """
        if menuObject in self._pluginMenus:
            # show all actions and activate their shortcuts
            for action in menuObject.actions() + [menuObject.menuAction()]:
                action.setVisible(True)
            
    def hidePluginMenus(self):
        """ Hides all menus in _pluginMenus list.
        """
        for menuObject in self._pluginMenus:
            # hide all actions and deactivate their shortcuts
            for action in menuObject.actions() + [menuObject.menuAction()]:
                action.setVisible(False)
    
    def createPluginToolBar(self, name):
        """ Creates tool bar in main window and adds it to _pluginToolBars list.
        """
        toolBar = self._window.addToolBar(name)
        self._pluginToolBars.append(toolBar)
        return toolBar

    def showPluginToolBar(self, toolBarObject):
        """ Shows given toolbar if it is in _pluginToolBars list.
        """
        if toolBarObject in self._pluginToolBars:
            toolBarObject.show()
            
    def hidePluginToolBars(self):
        """ Hides all toolbars in _toolBarMenus list.
        """
        for toolBar in self._pluginToolBars:
            toolBar.hide()
            
    def createZoomToolBar(self):
        """ Creates tool bar with 3 buttons "user", "100 %" and "all".
        
        See TabController's documentation of zoomUser(), zoomHundred() and zoomAll() to find out more on the different zoom levels.
        """
        self._zoomToolBar = self.createPluginToolBar('Zoom ToolBar')
        self._zoomToolBar.addAction(self.createAction('User', self.zoomUserEvent, image='zoomuser'))
        self._zoomToolBar.addAction(self.createAction('100 %', self.zoomHundredEvent, image='zoom100'))
        self._zoomToolBar.addAction(self.createAction('All', self.zoomAllEvent, image='zoomall'))
    
    def showZoomToolBar(self):
        """ Makes zoom tool bar visible.
        
        Should be called from TabController's selected() function, if the controller wants to use the tool bar.
        """
        self.showPluginToolBar(self._zoomToolBar)
    
    def hideZoomToolBar(self):
        """ Makes zoom tool bar invisible.
        """
        self._zoomToolBar.hide()
    
    def clearRecentFiles(self):
        """ Empties list of recent files and updates main menu.
        """ 
        self._recentFiles = []
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
       
    def getLastOpenLocation(self):
        """ Returns directory name of first entry of recent files list.
        """
        if len(self._recentFiles) > 0:
            return os.path.dirname(self._recentFiles[0])
        elif sys.platform == "darwin":
            # Mac OS X
            return homeDirectory + "/Documents"
        return homeDirectory
    
    def recentFiles(self):
        """ Returns list of recently opened files.
        """
        return self._recentFiles    
    
    def updateMenu(self):
        """ Update recent files and enable disable menu entries in file and edit menu.
        """
        logging.debug('Application: updateMenu()')
        # Recent files
        numRecentFiles = min(len(self._recentFiles), self.MAX_RECENT_FILES)
        for i in range(0, numRecentFiles):
            filename = self._recentFiles[i]
            self._recentFilesMenuActions[i].setText(os.path.basename(filename))
            self._recentFilesMenuActions[i].setToolTip(filename)
            self._recentFilesMenuActions[i].setStatusTip(filename)
            self._recentFilesMenuActions[i].setData(QVariant(filename))
            self._recentFilesMenuActions[i].setVisible(True)
            
        for i in range(numRecentFiles, self.MAX_RECENT_FILES):
            self._recentFilesMenuActions[i].setVisible(False)
            
        if numRecentFiles == 0:
            self._fileMenuItems['clearRecentFilesAction'].setEnabled(False)
        else:
            self._fileMenuItems['clearRecentFilesAction'].setEnabled(True)
            
        # Enabled / disable menu entries depending on number of open files
        atLeastOneFlag = False
        atLeastTwoFlag = False
        if self._window.tabWidget().count() > 1:
            atLeastOneFlag = True
            atLeastTwoFlag = True
        elif self._window.tabWidget().count() > 0:
            atLeastOneFlag = True        
        
        self._fileMenuItems['saveFileAction'].setEnabled(atLeastOneFlag)
        self._fileMenuItems['saveFileAsAction'].setEnabled(atLeastOneFlag)
        self._fileMenuItems['reloadFileAction'].setEnabled(atLeastOneFlag)
        self._fileMenuItems['closeFileAction'].setEnabled(atLeastOneFlag)
        
        self._fileMenuItems['saveAllFilesAction'].setEnabled(atLeastTwoFlag)
        self._fileMenuItems['closeAllAction'].setEnabled(atLeastTwoFlag)
        
        if atLeastOneFlag:
            try:
                if not self.currentTabController().isEditable():
                    self._fileMenuItems['saveFileAction'].setEnabled(False)
                    self._fileMenuItems['saveFileAsAction'].setEnabled(False)
                if not self.currentTabController().isModified():
                    self._fileMenuItems['saveFileAction'].setEnabled(False)
                
                copyPasteEnabled = self.currentTabController().isCopyPasteEnabled()
                self._editMenuItems['cutAction'].setEnabled(copyPasteEnabled)
                self._editMenuItems['copyAction'].setEnabled(copyPasteEnabled)
                self._editMenuItems['pasteAction'].setEnabled(copyPasteEnabled)
            
                self._editMenuItems['findAction'].setEnabled(self.currentTabController().isFindEnabled())
            except NoCurrentTabControllerException:
                pass
            
    def createAction(self, name, slot=None, shortcut=None, image=None):
        """ create an action with name and icon and connect it to a slot.
        """
        logging.debug('Application: createAction() - ' + name)
        if image:
            image0 = QPixmap()
            image0.load(":/resources/" + image + ".svg")
            #image0.load(os.path.join(resourceDirectory, image) + ".png")
            action = QAction(QIcon(image0), name, self._window)
        else:
            action = QAction(name, self._window)
        if slot:
            self.connect(action, SIGNAL("triggered()"), slot)
        if shortcut:
            if isinstance(shortcut, list):
                action.setShortcuts(shortcut)
            else:
                action.setShortcut(shortcut)
        return action
        
    def exit(self):
        self._window.close()

    def _collectFileExtensions(self):
        """ Loop over all plugins and remember their file extensions.
        """
        self._knownExtensionsDictionary = {}
        self._knownFiltersList = []
        self._knownFiltersList.append('All files (*.*)')
        for plugin in self.plugins():
            for ft in plugin.filetypes():
                self._knownExtensionsDictionary[ft.getExtension()] = plugin
                self._knownFiltersList.append(ft.getFileDialogFilter())
        if len(self._knownFiltersList) > 0:
            allKnownFilter = 'All known files (*.' + " *.".join(self._knownExtensionsDictionary.keys()) + ')'
            self._knownFiltersList.insert(1, allKnownFilter)
            logging.debug('Application: _collectFileExtensions() - ' + allKnownFilter)
        else:
            logging.debug('Application: _collectFileExtensions()')
        
        
    def openFileDialog(self):
        """Displays a common open dialog for all known file types.
        """
        logging.debug('Application: openFileDialog()')
        
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
        for i in range(0, self._window.tabWidget().count()):
            if filename == self._window.tabWidget().widget(i).controller().filename():
                self._window.tabWidget().setCurrentIndex(i)
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
        initCount = self._window.tabWidget().count()
        try:
            while self._window.tabWidget().count() > 0:
                if self.currentTabController().close() == False:
                    break
        except NoCurrentTabControllerException:
            pass
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
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
        
    def saveFileAsDialog(self):
        """This functions asks the user for a file name. 
        """
        logging.debug('Application: saveFileAsDialog()')
        try:
            currentTabController = self.currentTabController()
        except NoCurrentTabControllerException:
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
            return

        if currentTabController.filename():
            startDirectory = currentTabController.filename()
        else:
            startDirectory = self.getLastOpenLocation()
                
        filetypesList = []
        for filetype in currentTabController.supportedFileTypes():
            filetypesList.append(Filetype(filetype[0], filetype[1]).getFileDialogFilter())
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
        logging.debug('Application: savellFiles()')
        
        for i in range(0, self._window.tabWidget().count()):
            self._window.tabWidget().widget(i).tabController().save()

    def cutEvent(self):
        """ Called when cut action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        #if self._window.tabWidget().count() > 0:
        try:
            self.currentTabController().cut()
        except NoCurrentTabControllerException:
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
        
    def copyEvent(self):
        """ Called when copy action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        try:
            self.currentTabController().copy()
        except NoCurrentTabControllerException:
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
        
    def pasteEvent(self):
        """ Called when paste action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        try:
            self.currentTabController().paste()
        except NoCurrentTabControllerException:
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
            
    def findEvent(self):
        """ Called when find action is triggered (e.g. from menu entry) and forwards it to current tab controller.
        """
        try:
            self.currentTabController().find()
        except NoCurrentTabControllerException:
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
       
    def zoomUserEvent(self):
        """ Handles button pressed event from zoom tool bar and forwards it to current tab controller.
        """
        try:
            self.currentTabController().zoomUser()
        except NoCurrentTabControllerException:
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
        
    def zoomHundredEvent(self):
        """ Handles button pressed event  from zoom tool bar and forwards it to current tab controller.
        """
        try:
            self.currentTabController().zoomHundred()
        except NoCurrentTabControllerException:
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
    
    def zoomAllEvent(self):
        """ Handles button pressed event  from zoom tool bar and forwards it to current tab controller.
        """
        try:
            self.currentTabController().zoomAll()
        except NoCurrentTabControllerException:
            loggin.warning(self.__class__.__name__ + ": " + self.TAB_PREMATURELY_CLOSED_WARNING)
         
    def aboutBox(self):
        """ Displays about box. 
        """
        logging.debug('Application: aboutBox()')
        text = self._windowTitle()
        if self._version:
            text += " - " + str(self._version)
        # TODO: add vispaWebsiteUrl
        QMessageBox.about(self.mainWindow(), "About this software...", text) 

    def openRecentFile(self):
        """ Slot for opening recent file.
        
        Called from recent file menu action. Filename is set as data object (QVariant) of action.
        """
        filename = self.sender().data().toString()
        logging.debug('Application: openRecentFile() - ' + filename)
        self.openFile(filename)

    def tabChanged(self, tab=None):
        """ when a different tab is selected update menu
        """
        logging.debug('Application: tabChanged()')
        # only update once when closing all files at once
        if not self._closeAllFlag:
            self.hidePluginMenus()
            self.hidePluginToolBars()
            self.updateWindowTitle()
            self.updateMenu()
            try:
                self.currentTabController().selected()
                self.currentTabController().checkModificationTimestamp()
            except NoCurrentTabControllerException:
                pass        
    
    def currentFileModified(self):
        """ Update menu and window title.
        """
        self.updateMenu()
        self.updateWindowTitle()

    def _windowTitle(self):
        return str(self._window.windowTitle()).split("-")[0].strip()
    
    def updateWindowTitle(self):
        """ update window caption
        """
        logging.debug('Application: updateWindowTitle()')
        name = self._windowTitle()
        
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
        except IOError:
            configfile = open(os.path.abspath("Vispa.ini"), "w")
        self._ini.write(configfile)
        
    def _loadIni(self):
        """ Save the list of recent files.
        """
        logging.debug('Application: _loadIni()')
        ini = self.ini()
        if ini.has_option("history", "recentfiles"):
            text = str(ini.get("history", "recentfiles"))
            self._recentFiles = text.strip("[']").replace("', '", ",").split(",")
            if self._recentFiles == [""]:
                self._recentFiles = []
               
    def _saveIni(self):
        """ Load the list of recent files.
        """
        logging.debug('Application: _saveIni()')
        ini = self.ini()
        if not ini.has_section("history"):
            ini.add_section("history")
        ini.set("history", "recentfiles", str(self._recentFiles))
        self.writeIni()

    def errorMessage(self, message):
        """ Displays error message.
        """
        QMessageBox.critical(self.mainWindow(), 'Error', message)
        
    def warningMessage(self, message):
        """ Displays warning message.
        """
        QMessageBox.warning(self.mainWindow(), 'Warning', message)
        
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
            if hasattr(os, "startfile"):
                # Win32
                os.startfile(filename)
            elif sys.platform == "darwin":
                # OS X
                subprocess.call(("open", filename))
            elif sys.platform.find('linux') > - 1:
                # Linux
                try:
                  subprocess.call(("xdg-open", filename))
                except:
                  try:
                    subprocess.call(("gnome-open", filename))
                  except:
                    logging.error(self.__class__.__name__ + ": doubleClickOnFile() - Platform '" + sys.platform + "'. Cannot open file. I Don't know how!")
            else:
                logging.error(self.__class__.__name__ + ": doubleClickOnFile() - Unknown platform '" + sys.platform + "'. Cannot open file.")
        except:
            logging.error(self.__class__.__name__ + ": doubleClickOnFile() - Platform '" + sys.platform + "'. Error while opening file: " + str(filename))

    def createStatusBar(self):
        self._workingMessages = {}
        
        self._progressWidget = QLabel()
        self._window.statusBar().addPermanentWidget(self._progressWidget)

        self._progressTimeLine = QTimeLine(1000, self)
        self._progressTimeLine.setFrameRange(0, 100)
        self._progressTimeLine.setLoopCount(0)
        self.connect(self._progressTimeLine, SIGNAL("frameChanged(int)"), self.setProgress)
        
    def setProgress(self, progress):
        angle = int(progress * 360.0 / 100.0)
        pixmap = QPixmap(":/resources/vispabutton.png")
        rotate_matrix = QMatrix()
        rotate_matrix.rotate(angle)
        pixmap_rotated = pixmap.transformed(rotate_matrix)
        pixmap_moved = QPixmap(pixmap.size())
        pixmap_moved.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(pixmap_moved)
        painter.drawPixmap((pixmap_moved.width() - pixmap_rotated.width()) / 2.0, (pixmap_moved.height() - pixmap_rotated.height()) / 2.0, pixmap_rotated)
        painter.end()
        self._progressWidget.setPixmap(pixmap_moved.scaled(15, 15))
        
    def startWorking(self, message=""):
        self._window.statusBar().showMessage(message + "...")
        id = len(self._workingMessages.keys())
        self._workingMessages[id] = message
        if id == 0:
            self._progressTimeLine.start()
        return id

    def stopWorking(self, id, end="done"):
        self._window.statusBar().showMessage(self._workingMessages[id] + "..." + end + ".")
        del self._workingMessages[id]
        if len(self._workingMessages.keys()) == 0:
            self._progressTimeLine.stop()
