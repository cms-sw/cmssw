import logging
import os
import math

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.QtSvg import QSvgRenderer, QSvgWidget

from Vispa.Gui.VispaWidget import VispaWidget

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
        self.setCentralWidget(self._tabWidget)
        if hasattr(self._tabWidget,"setTabsClosable"):
            self._tabWidget.setTabsClosable(True)
        
        if "vispa" in title.lower():
            self.createStartupScreen()
            self.updateStartupScreenGeometry()

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
        #self._startupScreen.move(0.5 * (self.width() - self._startupScreen.width()), 0.5 * (self.height() - self._startupScreen.height()))

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
        
#    def paintEvent(self, event):
#        QMainWindow.paintEvent(self, event)
#        if self._startupScreen:
#            painter = QPainter(self)
#            painter.drawRect(self._startupScreen.boundingRect())

        
class StartupScreen(VispaWidget):
    
    # inherited parameters
    BACKGROUND_SHAPE = 'ROUNDRECT'
    #BACKGROUND_SHAPE = 'RECT'
    #BACKGROUND_SHAPE = 'CIRCLE'
    SELECTABLE_FLAG = False
    AUTOSIZE = True
    AUTOSIZE_KEEP_ASPECT_RATIO = False
    #ARROW_SHAPE = VispaWidget.ARROW_SHAPE_TOP
    #ARROW_SHAPE = VispaWidget.ARROW_SHAPE_RIGHT
    #ARROW_SHAPE = VispaWidget.ARROW_SHAPE_LEFT
    #ARROW_SHAPE = VispaWidget.ARROW_SHAPE_BOTTOM
    
    PROTOTYPING_DESCRIPTION = """Prototyping"""
    
    EXECUTING_DESCRIPTION = """Executing"""
    
    VERIFYING_DESCRIPTION = """Verifying"""
        
    def __init__(self, parent):
        self._descriptionWidgets = []
        self._descriptionActiveRects = [QRect(), QRect(), QRect()]   # descriptions will be visible if mouse cursor is in the rect
        VispaWidget.__init__(self, parent)
        self._filenewIcon = QIcon(QPixmap(":/resources/filenew.svg"))
        self._fileopenIcon = QIcon(QPixmap(":/resources/fileopen.svg"))
        self.setImage(QSvgRenderer(":/resources/startup_development_cycle.svg"))
        #self.setBodyWidget(QSvgWidget(":/resources/startup_development_cycle.svg"))
        self.setDragable(False)
        self.setMouseTracking(True)     # receive mouse events even if no button is pressed
        self._hideDescriptions = False
        
        
        self.createPrototypingWidget()
        self.createExecutionWidget()
        self.createVerifyingWidget()
        #self._prototypingRect = self._descriptionActiveRects[0]
        #self._executionRect = self._descriptionActiveRects[1]
        #self._verifyingRect = self._descriptionActiveRects[2]
        
    def createDescriptionWidget(self, arrowDirection, description):
        widget = VispaWidget(self.parent())
        #self._prototypingDescriptionWidget.setShape("ROUNDRECT")
        widget.enableAutosizing(True, False)
        widget.setSelectable(False)
        #widget.setText(description)
        #widget.setTitle(description)
        #widget.enableColorHeaderBackground(False)
        #widget.titleField().setPenColor(QColor(Qt.black))
        #widget.titleField().setDefaultFontSize(24)
        widget.setArrowShape(arrowDirection)
        widget.setVisible(not self._hideDescriptions)
        widget.setDragable(False)
        
        self._descriptionWidgets.append(widget)
        #self._descriptionActiveRects.append(rect)
        return widget
    
    def createPrototypingWidget(self):
        self._prototypingDescriptionWidget = self.createDescriptionWidget(VispaWidget.ARROW_SHAPE_BOTTOM, self.PROTOTYPING_DESCRIPTION)
        
        bodyWidget = QWidget(self._prototypingDescriptionWidget)
        bodyWidget.setLayout(QGridLayout())
        bodyWidget.layout().setContentsMargins(0, 0, 0, 0)
        
        bodyWidget.layout().addWidget(QLabel("Design physics analysis:"), 0, 0)
        analysisDesignerButton = QToolButton()
        analysisDesignerButton.setText("Analysis Designer")
        analysisDesignerButton.setIcon(self._filenewIcon)
        self.connect(analysisDesignerButton, SIGNAL("clicked(bool)"), self.parent().newAnalysisDesignerSlot)
        bodyWidget.layout().addWidget(analysisDesignerButton, 0, 1)
        bodyWidget.layout().addWidget(QLabel("Create physics event:"), 1, 0)
        pxlButton = QToolButton()
        pxlButton.setText("PXL Editor")
        pxlButton.setIcon(self._filenewIcon)
        self.connect(pxlButton, SIGNAL("clicked(bool)"), self.parent().newPxlSlot)
        bodyWidget.layout().addWidget(pxlButton, 1, 1)
    
        self._prototypingDescriptionWidget.setBodyWidget(bodyWidget)
        
    def createExecutionWidget(self):
        self._executionDescriptionWidget = self.createDescriptionWidget(VispaWidget.ARROW_SHAPE_RIGHT, self.EXECUTING_DESCRIPTION)
        
        bodyWidget = QWidget(self._executionDescriptionWidget)
        bodyWidget.setLayout(QGridLayout())
        bodyWidget.layout().setContentsMargins(0, 0, 0, 0)
        
        label=QLabel("Open and run existing analysis:")
        bodyWidget.layout().addWidget(label, 0, 0)
        analysisDesignerButton = QToolButton()
        analysisDesignerButton.setText("Open analysis file")
        analysisDesignerButton.setIcon(self._fileopenIcon)
        self.connect(analysisDesignerButton, SIGNAL("clicked(bool)"), self.parent().openAnalysisFileSlot)
        bodyWidget.layout().addWidget(analysisDesignerButton, 0, 1)
        self._analysisDesignerRecentFilesList=QListWidget()
        self._analysisDesignerRecentFilesList.setFixedSize(label.sizeHint().width()+analysisDesignerButton.sizeHint().width(),150)
        self.connect(self._analysisDesignerRecentFilesList, SIGNAL("doubleClicked(QModelIndex)"), self.parent().openAnalysisFileSlot)
        bodyWidget.layout().addWidget(self._analysisDesignerRecentFilesList, 1, 0, 1, 2)
        
        self._executionDescriptionWidget.setBodyWidget(bodyWidget)

    def analysisDesignerRecentFilesList(self):
        return self._analysisDesignerRecentFilesList
        
    def createVerifyingWidget(self):
        self._verifyingDescriptionWidget = self.createDescriptionWidget(VispaWidget.ARROW_SHAPE_LEFT, self.VERIFYING_DESCRIPTION)
        
        bodyWidget = QWidget(self._verifyingDescriptionWidget)
        bodyWidget.setLayout(QGridLayout())
        bodyWidget.layout().setContentsMargins(0, 0, 0, 0)
        
        label=QLabel("Browse an existing PXL data file:")
        bodyWidget.layout().addWidget(label, 0, 0)
        analysisDesignerButton = QToolButton()
        analysisDesignerButton.setText("Open PXL file")
        analysisDesignerButton.setIcon(self._fileopenIcon)
        self.connect(analysisDesignerButton, SIGNAL("clicked(bool)"), self.parent().openPxlFileSlot)
        bodyWidget.layout().addWidget(analysisDesignerButton, 0, 1)
        self._pxlEditorRecentFilesList=QListWidget()
        self._pxlEditorRecentFilesList.setFixedSize(label.sizeHint().width()+analysisDesignerButton.sizeHint().width(),150)
        self.connect(self._pxlEditorRecentFilesList, SIGNAL("doubleClicked(QModelIndex)"), self.parent().openPxlFileSlot)
        bodyWidget.layout().addWidget(self._pxlEditorRecentFilesList, 1, 0, 1, 2)
        
        self._verifyingDescriptionWidget.setBodyWidget(bodyWidget)
        
    def pxlEditorRecentFilesList(self):
        return self._pxlEditorRecentFilesList
        
#    def paint(self, painter):
#        #painter.drawRect(self.imageRectF())
#        VispaWidget.paint(self, painter)
#        painter.drawRect(self._executionRect)
#        painter.drawRect(self._verifyingRect)

    def mouseMoveEvent(self, event):
        if bool(event.buttons()):
            VispaWidget.mouseMoveEvent(self, event)
        elif self._hideDescriptions:
            for i in range(len(self._descriptionWidgets)):
                self._descriptionWidgets[i].setVisible(self._descriptionActiveRects[i].contains(event.pos()))
                
    def moveEvent(self, event):
        VispaWidget.moveEvent(self, event)
        self.rearangeDescriptionWidgets()
        
    def rearangeContent(self):
        VispaWidget.rearangeContent(self)
        self.rearangeDescriptionWidgets()
        
    def rearangeDescriptionWidgets(self):
        #self._activeSize = QSize(self._prototypingDescriptionWidget.getDistance("titleFieldWidth"), self._prototypingDescriptionWidget.getDistance("titleFieldHeight"))
        self._activeSize = QSize(0.3 * self.width(), 0.1 * self.height())
        self._prototypingRect = QRect(QPoint(0.5 * (self.width() - self._activeSize.width()), 0), self._activeSize)
        self._executionRect = QRect(QPoint(0, 0.635 * self.height()), self._activeSize)
        self._verifyingRect = QRect(QPoint(self.width() -self._activeSize.width(), 0.635 * self.height()), self._activeSize)
        self._descriptionActiveRects[0] = self._prototypingRect
        self._descriptionActiveRects[1] = self._executionRect 
        self._descriptionActiveRects[2] = self._verifyingRect
        
        self._prototypingDescriptionWidget.move(self.mapToParent(self._prototypingRect.topLeft()) + QPoint((self._prototypingRect.width() - self._prototypingDescriptionWidget.width()) * 0.5, - self._prototypingDescriptionWidget.height()))
        self._executionDescriptionWidget.move(self.mapToParent(self._executionRect.topLeft()) - QPoint(self._executionDescriptionWidget.width(), - 0.5 * (self._executionRect.height() - self._executionDescriptionWidget.height())))
        self._verifyingDescriptionWidget.move(self.mapToParent(self._verifyingRect.topRight()) - QPoint(0, - 0.5 * (self._verifyingRect.height() - self._verifyingDescriptionWidget.height())))
        
    def boundingRect(self):
        br = VispaWidget.boundingRect(self)
        for w in self._descriptionWidgets:
            br = br.united(w.boundingRect())
        return br

    def setVisible(self, visible):
        VispaWidget.setVisible(self, visible)
        self._executionDescriptionWidget.setVisible(visible and not self._hideDescriptions)
        self._prototypingDescriptionWidget.setVisible(visible and not self._hideDescriptions)
        self._verifyingDescriptionWidget.setVisible(visible and not self._hideDescriptions)
