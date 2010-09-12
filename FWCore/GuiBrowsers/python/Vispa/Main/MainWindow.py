import logging
import os
import math

from PyQt4.QtCore import Qt, SIGNAL, QEvent, QPoint, QSize
from PyQt4.QtGui import QMainWindow, QTabWidget, QSizePolicy, QIcon

from Vispa.Main.StartupScreen import StartupScreen
import Resources

class MainWindow(QMainWindow):
    
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    
    """ MainWindow """
    def __init__(self, application=None, title="VISPA"):
        #logging.debug(__name__ + ": __init__")
                
        self._justActivated = False
        self._startupScreen = None
        self._application = application
        QMainWindow.__init__(self)
        
        self._tabWidget = QTabWidget(self)
        self._tabWidget.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self._tabWidget.setUsesScrollButtons(True)
        self.setCentralWidget(self._tabWidget)
        if hasattr(self._tabWidget,"setTabsClosable"):
            self._tabWidget.setTabsClosable(True)
        
        if "vispa" in title.lower():
            self.createStartupScreen()

        self._fileMenu = self.menuBar().addMenu("&File")
        self._editMenu = self.menuBar().addMenu("&Edit")
        self._helpMenu = self.menuBar().addMenu("&Help")

        self._fileToolBar = self.addToolBar("File")
        
        self.ensurePolished()
        self.setWindowIcon(QIcon(":/resources/vispabutton.png"))
        self.setWindowTitle(title)
        self.statusBar()

        self._loadIni()
        if self._startupScreen:
            self._startupScreen.raise_()
            self.updateStartupScreenGeometry()
        
    def startupScreen(self):
        return self._startupScreen
        
    def application(self):
        return self._application
    
    def fileMenu(self):
        return self._fileMenu

    def editMenu(self):
        return self._editMenu
    
    def helpMenu(self):
        return self._helpMenu
    
    def fileToolBar(self):
        return self._fileToolBar
    
    def closeEvent(self, event):
        """ Closes all tabs and exits program if succeeded.
        """
        logging.debug('MainWindow: closeEvent()')
        self._application.closeAllFiles()
        self._application.shutdownPlugins()
        if len(self.application().tabControllers()) == 0:
            event.accept()
            self._saveIni()
        else:
            event.ignore()

    def addWindow(self, widget, width=None, height=None):
        """ Add a new window and call the TabController to update the label of the window.
        """
        logging.debug('MainWindow: addWindow()')
        widget.setMainWindow(self)
        widget.setWindowFlags(Qt.Dialog)
        widget.show()
        if width and height:
            widget.resize(width,height)
        else:
            widget.resize(self._tabWidget.size())
        widget.controller().updateLabel()

    def addTab(self, widget):
        """ Add a new tab to the TabWidget and call the TabController to update the label of the Tab.
        """
        #logging.debug('MainWindow: addTab()')
        widget.setTabWidget(self._tabWidget)
        widget.setMainWindow(self)
        self._tabWidget.addTab(widget, '')
        self._tabWidget.setCurrentWidget(widget)
        widget.controller().updateLabel()

    def tabWidget(self):
        return self._tabWidget
    
    def tabWidgets(self):
        return [self._tabWidget.widget(i) for i in range(0, self._tabWidget.count())]
    
    def isTabWidget(self, widget):
        return (self._tabWidget.indexOf(widget) >= 0)

    def _loadIni(self):
        """ Load the window properties.
        """
        ini = self._application.ini()
        
        if ini.has_option("window", "width"):
            width = ini.getint("window", "width")
        else:
            width = self.WINDOW_WIDTH
        if ini.has_option("window", "height"):
            height = ini.getint("window", "height")
        else:
            height = self.WINDOW_HEIGHT
        self.resize(QSize(width, height))
        if ini.has_option("window", "maximized"):
            if ini.getboolean("window", "maximized"):
                self.setWindowState(Qt.WindowMaximized)
        if ini.has_option("window", "fullScreen"):
            if ini.getboolean("window", "fullScreen"):
                self.setWindowState(Qt.WindowFullScreen)

    def _saveIni(self):
        """ Save the window properties.
        """
        ini = self._application.ini()
        if not ini.has_section("window"):
            ini.add_section("window")
        if not self.isMaximized() and not self.isFullScreen():
            ini.set("window", "width", str(self.width()))
            ini.set("window", "height", str(self.height()))
        ini.set("window", "maximized", str(self.isMaximized()))
        ini.set("window", "fullScreen", str(self.isFullScreen()))
        self._application.writeIni()
        
    def event(self, event):
        """ Emits activated() signal if correct event occures and if correct changeEvent occured before.
        
        Also see changeEvent().
        The Application shall connect to windowActivated().
        """
        QMainWindow.event(self, event)
        if self._justActivated and event.type() == QEvent.LayoutRequest:
            self._justActivated = False
            self.emit(SIGNAL("windowActivated()"))
        elif event.type()==QEvent.WindowActivate:
            self.emit(SIGNAL("windowActivated()"))
        return False
        
    def changeEvent(self, event):
        """ Together with event() this function makes sure tabChanged() is called when the window is activated.
        """
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self._justActivated = True

    def keyPressEvent(self, event):
        """ On Escape cancel all running operations.
        """
        #logging.debug(__name__ + ": keyPressEvent")
        if event.key() == Qt.Key_Escape:
            self.application().cancel()
        QMainWindow.keyPressEvent(self, event)

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        self.updateStartupScreenGeometry()
                
    def setStartupScreenVisible(self, show):
        if self._startupScreen:
            self._startupScreen.setVisible(show)
            #logging.debug(self.__class__.__name__ +": setStartupScreenVisible() %d" % self._startupScreen.isVisible())
            if show:
                self.updateStartupScreenGeometry()
            
    def updateStartupScreenGeometry(self):
        if not self._startupScreen:
            return
        boundingRect = self._startupScreen.boundingRect()
        deltaWidth = self.width() - boundingRect.width() - 20
        deltaHeight = self.height() - boundingRect.height() - 80
                
        if deltaWidth != 0 or deltaHeight != 0:
            self._startupScreen.setMaximumSize(max(1, self._startupScreen.width() + deltaWidth), max(1, self._startupScreen.height() + deltaHeight))
            boundingRect = self._startupScreen.boundingRect()
        self._startupScreen.move(QPoint(0.5 * (self.width() - boundingRect.width()), 0.5 * (self.height() - boundingRect.height()) + 10) + self._startupScreen.pos() - boundingRect.topLeft()) 

    def createStartupScreen(self):
        self._startupScreen = StartupScreen(self)
        
    def newAnalysisDesignerSlot(self, checked=False):
        """ Creates new analysis designer tab if that plugin was loaded.
        """
        plugin = self.application().plugin("AnalysisDesigner")
        if plugin:
            plugin.newAnalysisDesignerTab()
        
    def newPxlSlot(self, checked=False):
        """ Creates new pxl tab if that plugin was loaded.
        """
        plugin = self.application().plugin("Pxl")
        if plugin:
            plugin.newFile()
            
    def openAnalysisFileSlot(self, checked=False):
        plugin = self.application().plugin("AnalysisDesigner")
        if plugin:
            currentRow=self._startupScreen._analysisDesignerRecentFilesList.currentRow()
            if currentRow!=0:
                files=self.application().recentFilesFromPlugin(plugin)
                if currentRow<=len(files):
                    self.application().openFile(files[currentRow-1])
            else:
                filetypes = plugin.filetypes()
                if len(filetypes) > 0:
                    self.application().openFileDialog(filetypes[0].fileDialogFilter())
        
    def openPxlFileSlot(self, checked=False):
        plugin = self.application().plugin("Pxl")
        if plugin:
            currentRow=self._startupScreen._pxlEditorRecentFilesList.currentRow()
            if currentRow!=0:
                files=self.application().recentFilesFromPlugin(plugin)
                if currentRow<=len(files):
                    self.application().openFile(files[currentRow-1])
            else:
                filetypes = plugin.filetypes()
                if len(filetypes) > 0:
                    self.application().openFileDialog(filetypes[0].fileDialogFilter())
