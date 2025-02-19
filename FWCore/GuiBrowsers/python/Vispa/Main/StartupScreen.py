import logging

from PyQt4.QtCore import SIGNAL,QRect,QSize,QPoint
from PyQt4.QtGui import QToolButton,QIcon,QPixmap,QGridLayout,QLabel,QListWidget,QWidget
from PyQt4.QtSvg import QSvgRenderer, QSvgWidget

from Vispa.Gui.VispaWidget import VispaWidget

import Resources

class StartupScreen(VispaWidget):
    
    # inherited parameters
    BACKGROUND_SHAPE = 'ROUNDRECT'
    SELECTABLE_FLAG = False
    AUTOSIZE = True
    AUTOSIZE_KEEP_ASPECT_RATIO = False
    
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
        self.setDragable(False)
        self.setMouseTracking(True)     # receive mouse events even if no button is pressed
        self._hideDescriptions = False
        
        self.createPrototypingWidget()
        self.createExecutionWidget()
        self.createVerifyingWidget()
        
    def createDescriptionWidget(self, arrowDirection, description):
        widget = VispaWidget(self.parent())
        widget.enableAutosizing(True, False)
        widget.setSelectable(False)
        widget.setArrowShape(arrowDirection)
        widget.setVisible(not self._hideDescriptions)
        widget.setDragable(False)
        self._descriptionWidgets.append(widget)
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
