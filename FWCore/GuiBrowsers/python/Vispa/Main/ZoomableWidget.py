import logging
import os.path

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Main.Zoomable import *

class ZoomableWidget(QWidget, Zoomable):
    
    def __init__(self, parent=None):
        """ Constructor
        """
        QWidget.__init__(self, parent)
        Zoomable.__init__(self)
        
        if isinstance(self.parent(), ZoomableWidget):
#            logging.debug(__name__ + " __init__() parent().zoom() " + str(self.parent().zoom()))
            self.setZoom(self.parent().zoom())
        
    def setZoom(self, zoom):
        """ Sets zoom of this widget and of it's children.
        """
        Zoomable.setZoom(self, zoom)
        
        for child in self.children():
            if isinstance(child, Zoomable):
                child.setZoom(zoom)
        self.update()
        #logging.debug(__name__ +" setZoom() "+ str(zoom))
        
    def exportPDF(self, application):
        """ Experimental export to pdf files.
        
        If you do not know what you are doing do not use this function.
        """
        pixmap = QPixmap.grabWidget(self)
        pixmap.save('/tmp/out.png', 'PNG')
        
        
        printer = QPrinter(QPrinter.ScreenResolution)
        printer.setOutputFileName('/tmp/out.pdf')
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setFullPage(True)
        printer.setPaperSize(QSizeF(self.size()), QPrinter.DevicePixel)
        #printer.setPaperSize(QPrinter.Custom)
        
        print 'start redirect'
        QPainter.setRedirected(self, printer)
        #application.sendEvent(self, QPaintEvent(self.childrenRegion()))
        #application.sendEvent(self, QPaintEvent(self.childrenRect()))

        #for child in self.children():
        #    application.sendEvent(child, QPaintEvent(child.childrenRegion()))

        QPainter.restoreRedirected(self)
        
        print 'end redirect'
        for child in self.children():
            if hasattr(child, 'exportPDF'):
                print "found exportPDF child"
                #child.exportPDF(printer, application, QPoint(0,0))
                
        #from VispaWidget import *
        #from ConnectableWidget import *
        
        w2 = ConnectableWidget(self)
        w2.setTitle('test module')
        w2.setTextField('bal')
        w2.addSinkPort('default')
        #w2.setShape('CIRCLE')
        QPainter.setRedirected(w2, printer, QPoint(-50, - 50))
        #QPainter.setRedirected(w2, printer)
        application.sendEvent(w2, QPaintEvent(w2.rect()))
        QPainter.restoreRedirected(w2)
        
        w1 = VispaWidget(self)
        QPainter.setRedirected(w1, printer)
        application.sendEvent(w1, QPaintEvent(w1.rect()))
        QPainter.restoreRedirected(w1)

    def exportImage(self, fileName=None):
        types=""
        imageFormats=[str(f) for f in QImageWriter.supportedImageFormats()]
        for ft in imageFormats:
            if types!="":
                types+=";;"
            types+=ft.upper()+" File (*."+ft.lower()+")"
        filter=QString("")
        if not fileName:
            defaultname = "."
            fileName = str(QFileDialog.getSaveFileName(self,"Save image...",defaultname,types,filter))
        if fileName!="":
            name=fileName
            ext=str(filter).split(" ")[0].lower()
            if os.path.splitext(fileName)[1].lower().strip(".") in imageFormats:
                name=os.path.splitext(fileName)[0]
                ext=os.path.splitext(fileName)[1].lower().strip(".")
            picture=QPixmap.grabWidget(self,0,0,self.childrenRect().width(),self.childrenRect().height())
            picture.save(name+"."+ext,ext)
