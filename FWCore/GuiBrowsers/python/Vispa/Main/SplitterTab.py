import logging

from PyQt4.QtCore import Qt, SIGNAL, QCoreApplication, QEvent, QSize
from PyQt4.QtGui import QSplitter, QColor, QPalette, QToolBar, QSizePolicy, QWidget, QFrame, QHeaderView, QStandardItemModel, QVBoxLayout

from Vispa.Main.AbstractTab import AbstractTab
from Vispa.Views.PropertyView import PropertyView

class SplitterTab(QSplitter, AbstractTab):
    """ A Tab with a QSplitter and a function to create the PropertyView.
    
    The Tab is split vertically and within horizontally. QWidgets can be added
    to verticalSplitter() and horizontalSplitter(). In the constructor
    one can choose whether the PropertyView shall be on top level or inside
    the horizontalSplitter().
    """
    def __init__(self, parent=None, topLevelPropertyView=False):
        logging.debug(__name__ + ": __init__")
        self._topLevelPropertyView=topLevelPropertyView
        AbstractTab.__init__(self)
        if self._topLevelPropertyView:
            QSplitter.__init__(self, Qt.Horizontal, parent)
            self._verticalSplitter = QSplitter(Qt.Vertical, self)
        else:
            QSplitter.__init__(self, Qt.Vertical, parent)
            self._verticalSplitter = self
        
        self._horizontalSplitter = QSplitter(Qt.Horizontal, self._verticalSplitter)
        
        self.connect(self._verticalSplitter, SIGNAL("splitterMoved(int, int)"), self.verticalSplitterMovedSlot)
        self.connect(self._horizontalSplitter, SIGNAL("splitterMoved(int, int)"), self.horizontalSplitterMovedSlot)
        if self._topLevelPropertyView:
            self.connect(self, SIGNAL("splitterMoved(int, int)"), self.horizontalSplitterMovedSlot)
        self._toolBar = None
        
        self._propertyView = None
        
    def verticalSplitter(self):
        return self._verticalSplitter
    
    def verticalSplitterMovedSlot(self, pos, index):
        """ Implement this function if you want to react on size changes invoked by the vertical splitter.
        """
        pass
    
    def horizontalSplitter(self):
        return self._horizontalSplitter
    
    def horizontalSplitterMovedSlot(self, pos, index):
        """ Implement this function if you want to react on size changes invoked by the horizontal splitter.
        """
        pass
    
    def createToolBar(self,index=None):
        if index==None:
            index = self.verticalSplitter().count()
        self._toolBar = SplitterToolBar()
        self.verticalSplitter().insertWidget(index,self._toolBar)
        self.verticalSplitter().setCollapsible(index, False)
        self._toolBar.setFixedHeight(20)
        self._toolBar.show()
        
    def toolBar(self):
        return self._toolBar 
        
    def setController(self, controller):
        AbstractTab.setController(self, controller)
        if self._propertyView:
            self._propertyView.setReadOnly(not self.controller().isEditable())
            self.connect(self._propertyView, SIGNAL('valueChanged'), self.controller().setModified)
        
    def createPropertyView(self):
        """ Creates PropertyView object, adds it to this tab and makes it available via propertyView().
        """
        if self._topLevelPropertyView:
            parent=self
        else:
            parent=self.horizontalSplitter()
        self._propertyView = PropertyView(parent, "PropertyView")
        
    def propertyView(self):
        """ Returns PropertyView object. See createPropertyView().
        """
        return self._propertyView
    
    def closeEvent(self,event):
        """ Call close if tab is not embedded in TabWidget
        """
        if not self._tabWidget and self.mainWindow().isTabWidget(self):
            if not self.controller().close():
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()

    def event(self,event):
        """ Call tabChanged if window is activated.
        """
        if not self._tabWidget:
            if event.type()==QEvent.WindowActivate:
                QCoreApplication.instance().tabChanged()
        return QSplitter.event(self,event)


class SplitterToolBar(QToolBar):
    """ A vertical toolbar which can be split up in sections.
    
    The toolbar can be placed in the verticalSplitter() of the SplitterTab
    to separate two parts. The toolbar can hold buttons to control its
    neighboring views.
    """
    
    ALIGNMENT_LEFT = 0
    ALIGNMENT_CENTER = 1
    ALIGNMENT_RIGHT = 2
    
    def __init__(self, parent=None):
        self._sections = []
        self._sizes = None
        QToolBar.__init__(self, parent)
        self.layout().setSpacing(0)
        
    def _createSpacer(self, width=0):
        spacer = QWidget(self)
        spacer.setFixedSize(width, self.iconSize().height())
        return spacer
        
    def addSection(self, alignment=None):
        if alignment == None:
            alignment = self.ALIGNMENT_CENTER
    
        leftSpacer = self._createSpacer()
        rightSpacer = self._createSpacer()
        contentList = []
        leftSpacerAction = self.addWidget(leftSpacer)
        rightSpacerAction = self.addWidget(rightSpacer)
        section = {"leftSpacer": leftSpacer, "leftSpacerAction": leftSpacerAction, 
                   "contentList": contentList, 
                   "rightSpacer": rightSpacer, "rightSpacerAction": rightSpacerAction,
                   "alignment": alignment}
        
        self._sections.append(section)
        if self._sizes:
            self.setSectionSizes(self._sizes)
        return len(self._sections) - 1
    
    def addWidgetToSection(self, widget, section):
        if section > len(self._sections) - 1:
            logging.error(self.__class__.__name__ +": addWidgetToSection() - Unknown section number %d. Aborting..." % section)
            return
        
        widget.show()
        self._sections[section]["contentList"].append(widget)
        self.insertWidget(self._sections[section]["rightSpacerAction"], widget)
        # TODO: implement remove function
        if self._sizes:
            self.setSectionSizes(self._sizes)
        
    def setSectionSizes(self, sizes):
        """ sizes is list of widths for section spacers.
        """
        self._sizes = sizes # try to fix design, need to find better solution
        for i in range(min(len(self._sections), len(sizes))):
            section = self._sections[i]
            contentsWidth = 0
            for content in section["contentList"]:
                contentsWidth += content.width() + self.layout().spacing() + 1
            if section["alignment"] == self.ALIGNMENT_LEFT:
                rightWidth = sizes[i] - contentsWidth
                leftWidth = 0
            elif section["alignment"] == self.ALIGNMENT_RIGHT:
                contentsWidth += self.layout().spacing() * 2
                rightWidth = 0
                leftWidth = sizes[i] - contentsWidth
            else:
                # Default is ALIGNMENT_CENTER
                leftWidth = 0.5 * (sizes[i] - contentsWidth)
                rightWidth = leftWidth
            section["leftSpacer"].setFixedSize(max(0, leftWidth), self.iconSize().height())
            section["rightSpacer"].setFixedSize(max(0, rightWidth), self.iconSize().height())
            
    def takeToolBoxContainerButtons(self, toolBox, section):
        for button in toolBox.toggleButtons():
            self.addWidgetToSection(button, section)
            
    def showEvent(self, event):
        QToolBar.showEvent(self, event)
        if self._sizes:
            self.setSectionSizes(self._sizes)