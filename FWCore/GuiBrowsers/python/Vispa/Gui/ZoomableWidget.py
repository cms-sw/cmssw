import logging
import os.path

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Gui.Zoomable import Zoomable
from Vispa.Share.ImageExporter import ImageExporter

class ZoomableWidget(QWidget, Zoomable):
    
    def __init__(self, parent=None):
        """ Constructor
        """
        QWidget.__init__(self, parent)
        Zoomable.__init__(self)
        self._imageExporter = None
        
        if isinstance(self.parent(), ZoomableWidget):
            self.setZoom(self.parent().zoom())
        
    def setZoom(self, zoom):
        """ Sets zoom of this widget and of it's children.
        """
        Zoomable.setZoom(self, zoom)
        
        for child in self.children():
            if isinstance(child, Zoomable):
                child.setZoom(zoom)
        self.update()
        
    def exportImage(self, filename=None):
        if not self._imageExporter:
            self._imageExporter = ImageExporter(self)
            
        if not filename:
            self._imageExporter.exportImageDialog(self)
        else:
            self._imageExporter.exportImage(self, filename)