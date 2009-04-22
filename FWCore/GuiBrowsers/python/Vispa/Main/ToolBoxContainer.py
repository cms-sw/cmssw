from PyQt4.QtCore import *
from PyQt4.QtGui import *

import logging

class ToolBoxContainer(QWidget):
    """ Container for widgets which can be shown or hidden by a row of toggle buttons at the bottom of the container
    
    The container uses a vertical QSplitter object to show added widgets.
    """
    
    def __init__(self, parent=None):
        """ Constructor.
        """
        logging.debug(__name__ +": __init__()")
        QWidget.__init__(self, parent)
        self._toolboxList = []
        self._toggleButtons = []
        self._splitter = QSplitter(Qt.Vertical)
        
        # initialize layout
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().addWidget(self._splitter, 3)
        
        self._buttonLayout = QHBoxLayout()
        self._buttonLayout.addStretch(2)      # push buttons completely to the right
        self._buttonLayout.setSpacing(0)
        
        self.layout().addStretch(0.5)             # keep buttons at the bottom even if all tool boxes are invisible
        self.layout().addLayout(self._buttonLayout)
        
    def splitter(self):
        """ Returns splitter containing widgets.
        """
        return self._splitter
        
    def addWidget(self, widget):
        """ Adds widget to tool box.
        """
        #self.layout().insertWidget(len(self._toolboxList), widget, 3)
        self._splitter.addWidget(widget)
        self._toolboxList.append(widget)
        toggleButton = QToolButton()
        toggleButton.setCheckable(True)
        toggleButton.setChecked(True)
        toggleButton.setText("v")
        toggleButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.connect(toggleButton, SIGNAL('clicked(bool)'), self.toggleButtonPressed)
        self._buttonLayout.addWidget(toggleButton, 0, Qt.AlignRight)
        self._toggleButtons.append(toggleButton)
        
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
            