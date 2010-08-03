import os.path

from PyQt4 import QtCore, QtGui

import logging

class ImageExporter(QtCore.QObject):
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        self._imageFormats = ["pdf", "ps"] + [str(f) for f in QtGui.QImageWriter.supportedImageFormats()]
        fileFilters = [self._fileFilter(ft) for ft in self._imageFormats]
        self._fileFiltersString = ";;".join(fileFilters)
        self._selectedFileFilter = fileFilters[0]
        self._exportImageFileName = None
        
    def _fileFilter(self, ext):
        return ext.upper() +" File (*."+ ext.lower() + ")"
        
    def exportPdfHelper(self, widget, painter, offset):
        """ This function recursively does the painting for exportPdf().
        """
        picture = QtGui.QPicture()
        QtGui.QPainter.setRedirected(widget, picture)
        QtGui.QApplication.instance().sendEvent(widget, QtGui.QPaintEvent(widget.rect()))
        QtGui.QPainter.restoreRedirected(widget)
        painter.drawPicture(offset, picture)
        
        for child in widget.children():
            if isinstance(child, QtGui.QWidget) and child.isVisible():
                self.exportPdfHelper(child, painter, offset + child.pos())
        
    def exportPdf(self, widget, filename, format=QtGui.QPrinter.PdfFormat):
        """ This function draws itself into a pdf file with given filename.
        
        format is any format type supported by QtGui.QPrinter.
        """
        logging.debug(self.__class__.__name__ +": exportPdf()")

        printer = QtGui.QPrinter(QtGui.QPrinter.ScreenResolution)
        printer.setOutputFileName(filename)
        printer.setOutputFormat(format)
        printer.setFullPage(True)
        printer.setPaperSize(QtCore.QSizeF(widget.size()), QtGui.QPrinter.DevicePixel)
        painter = QtGui.QPainter()
        painter.begin(printer)
        
        self.exportPdfHelper(widget, painter, QtCore.QPoint(0,0))
        painter.end()
        
    def exportImageDialog(self, widget):
        if self._exportImageFileName:
            defaultname = self._exportImageFileName
        else:
            defaultname = QtCore.QCoreApplication.instance().getLastOpenLocation()
            
        filter = QtCore.QString(self._selectedFileFilter)
        filename = str(QtGui.QFileDialog.getSaveFileName(self.parent(),"Save image...",defaultname, self._fileFiltersString, filter))
        ext = str(filter).split(" ")[0].lower() # extract ext from selected filter
        self.exportImage(widget, filename)
    
    def exportImage(self, widget, filename, ext=""):
        self._exportImageFileName = filename
        name = filename
        
        if os.path.splitext(filename)[1].lower().strip(".") in self._imageFormats:
            name=os.path.splitext(filename)[0]
            ext=os.path.splitext(filename)[1].lower().strip(".")
        
        self._selectedFileFilter = self._fileFilter(ext)
        if ext == "pdf":
            self.exportPdf(widget, name +"."+ ext, QtGui.QPrinter.PdfFormat)
        elif ext == "ps":
            self.exportPdf(widget, name +"."+ ext, QtGui.QPrinter.PostScriptFormat)
        else:
            childrenRect = widget.childrenRect()
            margin = 10
            picture = QtGui.QPixmap.grabWidget(widget, childrenRect.left() - margin, childrenRect.top() - margin, childrenRect.width() + 2*margin, childrenRect.height() + 2*margin)
        
        # grabWidget() does not work on Mac OS X with color gradients
        # grabWidget() grabs pixels directly from screen --> can lead to problems if window is partly covered
            picture.save(name+"."+ext,ext)
