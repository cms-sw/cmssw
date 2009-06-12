import logging
import os
import math

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import Resources

class MainWindow(QMainWindow):
    
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    
    """ MainWindow """
    def __init__(self, application=None, title="VISPA"):
        logging.debug(__name__ + ": __init__")
                
        self._justActivated = False
        QMainWindow.__init__(self)
        
        self._application = application

        self._fileMenu = self.menuBar().addMenu("&File")
        self._editMenu = self.menuBar().addMenu("&Edit")
        self._helpMenu = self.menuBar().addMenu("&Help")

        self._fileToolBar = self.addToolBar("File")
        
        self.ensurePolished()
        self.setWindowTitle(title)
        self.statusBar()

        self._tabWidget = QTabWidget(self)
        self._tabWidget.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.setCentralWidget(self._tabWidget)
        if hasattr(self._tabWidget,"setTabsClosable"):
            self._tabWidget.setTabsClosable(True)
       
        self.setWindowIcon(QIcon(":/resources/vispabutton.png"))
        
        self._loadIni()
        
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
        if self._tabWidget.count() == 0:
            event.accept()
            self._saveIni()
        else:
            event.ignore()

    def addTab(self, widget):
        """ Add a new tab to the TabWidget and call the TabController to update the label of the Tab.
        """
        logging.debug('MainWindow: addTab()')
        widget.setTabWidget(self._tabWidget)
        widget.setMainWindow(self)
        self._tabWidget.addTab(widget, '')
        self._tabWidget.setCurrentWidget(widget)
        widget.controller().updateLabel()

    def tabWidget(self):
        return self._tabWidget

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
        """
        QMainWindow.event(self, event)
        if self._justActivated and event.type() == QEvent.LayoutRequest:
            self._justActivated = False
            self.emit(SIGNAL("activated()"))
        return False
        
    def changeEvent(self, event):
        """ Together with event() this function makes sure tabChanged() is called when the window is activated.
        """
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self._justActivated = True
