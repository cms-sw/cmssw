import sys,os
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from PyQt4 import QtGui, QtCore

class LumiCanvas(FigureCanvas):
    """this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, fig=None):
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
class ApplicationWindow(QtGui.QMainWindow):
    '''
    main evt loop
    '''
    def __init__(self,fig=None):
        self.qApp=QtGui.QApplication(sys.argv)#every PyQt4 application must create an application object
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.main_widget = QtGui.QWidget(self)
        l = QtGui.QVBoxLayout(self.main_widget)
        sc = LumiCanvas(self.main_widget,fig=fig)
        l.addWidget(sc)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
    def fileQuit(self):
        self.close()
    def closeEvent(self, ce):
        self.fileQuit()
    def destroy(self):
        sys.exit(self.qApp.exec_())
        
if __name__ == "__main__":
    from numpy import arange, sin, pi
    from matplotlib.figure import Figure
    fig=Figure(figsize=(7.2,5.4),dpi=120)#create fig
    t = arange(0.0,3.0,0.01)
    s = sin(2*pi*t)
    ax=fig.add_subplot(111)
    ax.plot(t,s)
    aw=ApplicationWindow(fig=fig)
    aw.show()
    aw.destroy()
