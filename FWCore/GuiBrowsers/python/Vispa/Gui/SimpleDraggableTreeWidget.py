from PyQt4.QtCore import QMimeData, QByteArray, Qt, QSize, QPoint, QVariant,SIGNAL
from PyQt4.QtGui import QTreeWidget, QImage, QDrag, QPixmap, QIcon, QPalette, QColor,QTreeWidgetItem

import logging

class SimpleDraggableTreeWidget(QTreeWidget):
    """ TreeWidget suitable for holding a list of strings.
    """
    MIME_TYPE = "text/plain"
    def __init__(self, headerLabel, dragEnabled=False, mimeType=None, parent=None):
        """ Constructor.
        """
        QTreeWidget.__init__(self,parent)
        self.setMimeType(mimeType)
        self.setAutoFillBackground(True)
        #print "color roles", QPalette.Base, QPalette.Window, self.backgroundRole()
        lightBlueBackgroundColor = QColor(Qt.blue).lighter(195)
        #lightBlueBackgroundColor = QColor(Qt.red)
        self.palette().setColor(QPalette.Base, lightBlueBackgroundColor)       # OS X
        self.palette().setColor(QPalette.Window, lightBlueBackgroundColor)
        
        self.setColumnCount(1)
        self.setHeaderLabels([headerLabel])
        if dragEnabled:
            self.setDragEnabled(True)

    def populate(self, items):
        """ Fills items into list.
        """
        self.insertTopLevelItems(0, items)

    def setDragEnable(self, dragEnabled, mimeType=None):
        """ Usual behavior of QWidget's setDragEnabled() function plus optional setting of mimeType.
        """
        QTreeWidget.setDragEnabled(dragEnabled)
        self.setMimeType(mimeType)
        
    def mimeType(self):
        """ Returns mime type which will be used to encode list entries while dragging.
        """
        return self._mimeType
    
    def setMimeType(self, mimeType):
        """ Sets mime type of this widget to type if type is not None.
        
        If type is None the default mime type MIME_TYPE will be used.
        """
        if mimeType:
            self._mimeType = mimeType
        else:
            self._mimeType = self.MIME_TYPE
        
    def mimeTypes(self):
        """ Returns self.mimeType() as single element of QStringList.
        """
        list = QStringList()
        list << self.mimeType()
        return list
    
    def mimeData(self, items):
        """ Returns QMimeData for drag and drop.
        """
        logging.debug(self.__class__.__name__ + ": mimeData()")
        mime = QMimeData()
        encodedData = QByteArray()
        
        for item in items:
            encodedData.append(item.text(0))
        mime.setData(self.mimeType(), encodedData)
        return mime
    
    def startDrag(self, supportedActions):
        """ Overwritten function of QTreeWidget.
        
        This function creates a QDrag object representing the selected element of this TreeWidget.
        """
        logging.debug(self.__class__.__name__ +": startDrag()")
        indexes = self.selectedIndexes()
        if len(indexes) > 0:
            data = self.model().mimeData(indexes)
            if not data:
                return
            drag = QDrag(self)
            drag.setMimeData(data)
            if self.model().data(indexes[0], Qt.DecorationRole).type() == QVariant.Icon:
            	icon = QIcon(self.model().data(indexes[0], Qt.DecorationRole))
                drag.setPixmap(icon.pixmap(QSize(50, 50)))
                drag.setHotSpot(QPoint(drag.pixmap().width()/2, drag.pixmap().height()/2))  # center icon in respect to cursor
            defaultDropAction = Qt.IgnoreAction
            drag.exec_(supportedActions, defaultDropAction)

    def mousePressEvent(self,event):
        QTreeWidget.mousePressEvent(self,event)
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos())
    