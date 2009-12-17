import logging
import os.path

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Gui.Zoomable import Zoomable

class ZoomableWidget(QWidget, Zoomable):
    
    def __init__(self, parent=None):
        """ Constructor
        """
        QWidget.__init__(self, parent)
        Zoomable.__init__(self)
        self._exportImageFileName = None
        
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
    
    def exportPdfHelper(self, painter, offset):
        """ This function recursively does the painting for exportPdf().
        """
        picture = QPicture()
        QPainter.setRedirected(self, picture)
        QApplication.instance().sendEvent(self, QPaintEvent(self.rect()))
        QPainter.restoreRedirected(self)
        painter.drawPicture(offset, picture)
        
        for child in self.children():
            if hasattr(child, 'exportPdfHelper'):
                child.exportPdfHelper(painter, offset + child.pos())
        
    def exportPdf(self, filename, format=QPrinter.PdfFormat):
        """ This function draws itself into a pdf file with given filename.
        
        format is any format type supported by QPrinter.
        """
        logging.debug(self.__class__.__name__ +": exportPdf()")
         
        #filename = "/tmp/out.png"
        printer = QPrinter(QPrinter.ScreenResolution)
        printer.setOutputFileName(filename)
        printer.setOutputFormat(format)
        printer.setFullPage(True)
        printer.setPaperSize(QSizeF(self.size()), QPrinter.DevicePixel)
        painter = QPainter()
        painter.begin(printer)
        
        #image = QImage()    # does not work
        #image = QPixmap(self.size())
        #painter.begin(image)
        self.exportPdfHelper(painter, QPoint(0,0))
        painter.end()
        #image.save(filename)
        
    def exportImage(self, fileName=None):
        types=""
        imageFormats = ["pdf", "ps"] + [str(f) for f in QImageWriter.supportedImageFormats()]
        for ft in imageFormats:
            if types!="":
                types+=";;"
            types+=ft.upper()+" File (*."+ft.lower()+")"
        filter=QString("")
        if not fileName:
            if self._exportImageFileName:
                defaultname = self._exportImageFileName
            else:
                defaultname = QCoreApplication.instance().getLastOpenLocation()
            fileName = str(QFileDialog.getSaveFileName(self,"Save image...",defaultname,types,filter))
        if fileName!="":
            self._exportImageFileName = fileName
            name = fileName
            ext=str(filter).split(" ")[0].lower()
            
            if os.path.splitext(fileName)[1].lower().strip(".") in imageFormats:
                name=os.path.splitext(fileName)[0]
                ext=os.path.splitext(fileName)[1].lower().strip(".")
            #self.exportPdf("")
            #return
            if ext == "pdf":
                self.exportPdf(name +"."+ ext, QPrinter.PdfFormat)
            elif ext == "ps":
                self.exportPdf(name +"."+ ext, QPrinter.PostScriptFormat)
            else:
                childrenRect = self.childrenRect()
                margin = 10
                picture = QPixmap.grabWidget(self, childrenRect.left() - margin, childrenRect.top() - margin, childrenRect.width() + 2*margin, childrenRect.height() + 2*margin)
            
            # grabWidget() does not work on Mac OS X with color gradients
            # grabWidget() grabs pixels directly from screen --> can lead to problems if window is partly covered
                picture.save(name+"."+ext,ext)
