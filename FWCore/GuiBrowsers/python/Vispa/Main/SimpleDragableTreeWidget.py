from PyQt4.QtCore import QMimeData, QByteArray
from PyQt4.QtGui import QTreeWidget

import logging

class SimpleDragableTreeWidget(QTreeWidget):
    """ TreeWidget suitable for holding a list of strings.
    """
    MIME_TYPE = "text/plain"
    def __init__(self, mimeType=None):
        """ Constructor.
        """
        QTreeWidget.__init__(self)
        if mimeType:
            self._mimeType = mimeType
        else:
            self._mimeType = self.MIME_TYPE
        
    def mimeType(self):
        """ Returns mime type which will be used to encode list entries while dragging.
        """
        return self._mimeType
        
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
     