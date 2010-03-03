from PyQt4.QtCore import Qt,SIGNAL
from PyQt4.QtGui import QHeaderView, QStandardItemModel, QToolButton, QFrame, QVBoxLayout, QSizePolicy


class FrameWithHeader(QFrame):
    
    def __init__(self, parent=None):
        QFrame.__init__(self, parent)
        self.setFrameShadow(QFrame.Raised)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        
        self._header = Header(Qt.Horizontal, self)
        self.layout().addWidget(self._header)
        
    def addWidget(self, widget):
        if isinstance(widget, QFrame):
            widget.setFrameStyle(QFrame.NoFrame)
        self.layout().addWidget(widget)
        
    def header(self):
        return self._header

class Header(QHeaderView):
    
    def __init__(self, orientation, parent=None):
        QHeaderView.__init__(self, orientation, parent)
        
        self.setModel(QStandardItemModel(self))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStretchLastSection(True)
        self.setFixedHeight(25)
        self._menuButton = None
    
    def createMenuButton(self, label=">"):
        self._menuButton = QToolButton(self)
        self._menuButton.setText(label)
        return self._menuButton
        
    def menuButton(self):
        return self._menuButton
    
    def setText(self, text):
        if self.orientation() == Qt.Horizontal:
            self.model().setHorizontalHeaderLabels([text])
        elif self.orientation() == Qt.Vertical:
            self.model().setVerticalHeaderLabels([text])
    
    def mousePressEvent(self,event):
        QHeaderView.mousePressEvent(self,event)
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos())
