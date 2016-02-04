from PyQt4.QtCore import *
from PyQt4.QtGui import *

import logging

class ToolBoxContainer(QWidget):
    """ Container for widgets which can be shown or hidden by a row of toggle buttons at the bottom of the container
    
    The container uses a vertical QSplitter object to show added widgets.
    """
    
    HIDE_LAST_TOGGLE_BUTTON = True  # see setHideLastToggleButton()
    
    TYPE_SPLITTER = 0
    TYPE_WIDGETS = 2
    
    def __init__(self, parent=None, containerType=None):
        """ Constructor.
        """
        logging.debug(__name__ +": __init__()")
        QWidget.__init__(self, parent)
        self._hideLastToggleButton = False
        self._containerType = self.TYPE_SPLITTER
        self.setHideLastToggleButton(self.HIDE_LAST_TOGGLE_BUTTON)
        self._toolboxList = []
        self._toggleButtons = []
        self._splitter = None
        if containerType:
            self.setType(containerType)
            
        # initialize layout
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)
        if self._containerType == self.TYPE_SPLITTER:
            self._splitter = QSplitter(Qt.Vertical)
            self.layout().addWidget(self._splitter, 3)
        
        self._buttonLayout = QHBoxLayout()
        self._buttonLayout.addStretch(2)      # push buttons completely to the right
        self._buttonLayout.setSpacing(0)
        
        self.layout().addStretch(0.5)         # keep buttons at the bottom even if all tool boxes are invisible
        self.layout().addLayout(self._buttonLayout)
        
    def setType(self, containerType):
        self._containerType = containerType
        
    def setHideLastToggleButton(self, hide):
        """ Influences visibility of last visible toggle button.
        
        If hide is True toggle buttons are only shown if there is more than one widget selectable.
        In this case the ToolBoxContainer behaves like a normal widget. 
        If hide if False the toggle button is also show if there is only one widget selectable.
        """
        self._hideLastToggleButton = hide
        
    def splitter(self):
        """ Returns splitter containing widgets.
        """
        return self._splitter
    
    def toggleButtons(self):
        return self._toggleButtons
        
    def addWidget(self, widget, stretch=0):
        """ Adds widget to tool box.
        """
        if self._containerType == self.TYPE_SPLITTER:
            self._splitter.addWidget(widget)
        else:
            self.layout().insertWidget(len(self._toolboxList), widget, stretch)
        self._toolboxList.append(widget)
        toggleButton = QToolButton()
        toggleButton.setCheckable(True)
        toggleButton.setChecked(True)
        toggleButton.setText("v")
        toggleButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.connect(toggleButton, SIGNAL('clicked(bool)'), self.toggleButtonPressed)
        self._buttonLayout.addWidget(toggleButton, 0, Qt.AlignRight)
        self._toggleButtons.append(toggleButton)
        
        self.updateToggleButtonVisiblity()
        
    def showWidget(self, widget, show=True):
        if widget in self._toolboxList:
            widget.setVisible(show)
            self._toggleButtons[self._toolboxList.index(widget)].setVisible(show)
            self.updateToggleButtonVisiblity()
            
    def updateToggleButtonVisiblity(self):
        if self._hideLastToggleButton:
            indices = self.visibleToggleButtonsIdices()
            if len(indices) == 1:
                self._toggleButtons[indices[0]].hide()
        
        # make sure toggle buttons are shown if there are more than one widget selectable
        # or if hideLastToggleButton was changed in the meantime
        indices = self.visibleToolBoxIdices()
        if len(indices) > 1 or not self._hideLastToggleButton:
            for i in indices:
                if not self._toggleButtons[i].isVisible():
                    self._toggleButtons[i].show()
            
    def hideWidget(self, widget):
        self.showWidget(widget, False)
        
    def visibleToggleButtonsIdices(self):
        """ Returns list of indices of toggle buttons which are visible.
        """
        return self._visibleIndices(self._toggleButtons)
    
    def visibleToolBoxIdices(self):
        """ Returns list of indices of tool box widgets which are visible.
        """
        return self._visibleIndices(self._toolboxList)
    
    def _visibleIndices(self, list):
        """ Returns list of indices of entries in given list which are visible.
        
        It is assumed list entries have boolean function isVisible() (e.g. QWidget).
        """
        indices = []
        for i in range(len(list)):
            if list[i].isVisible():
                indices.append(i)
        return indices
        
    def toggleButtonPressed(self):
        """ Slot for showing and hinding widgets when toggle buttons are pressed.
        """
        if self.sender() in self._toggleButtons:
            index = self._toggleButtons.index(self.sender())
            
            if self._toolboxList[index].isVisible():
                self._toolboxList[index].hide()
                self.sender().setText("^")
            else:
                self._toolboxList[index].show()
                self.sender().setText("v")
            