from PyQt4.QtCore import QTimeLine,SIGNAL,Qt
from PyQt4.QtGui import QLabel,QPixmap,QMatrix,QPainter

class RotatingIcon(QLabel):
    def __init__(self,resource,parent=None,steps=20,width=15,height=15):
        QLabel.__init__(self,parent)
        self._resource=resource
        self._steps=steps
        self._width=width
        self._height=height
        self._progressTimeLine = QTimeLine(1000, self)
        self._progressTimeLine.setFrameRange(0, self._steps)
        self._progressTimeLine.setLoopCount(0)
        self.connect(self._progressTimeLine, SIGNAL("frameChanged(int)"), self.setProgress)
        self._renderPixmaps()
        self.setProgress(0)

    def _renderPixmaps(self):
        self._pixmaps=[]
        for i in range(self._steps+1):
            angle = int(i * 360.0 / self._steps)
            pixmap = QPixmap(self._resource)
            # if problem with loading png
            if pixmap.size().width()==0:
                self._pixmaps=None
                return
            rotate_matrix = QMatrix()
            rotate_matrix.rotate(angle)
            pixmap_rotated = pixmap.transformed(rotate_matrix)
            pixmap_moved = QPixmap(pixmap.size())
            pixmap_moved.fill(Qt.transparent)
            painter = QPainter()
            painter.begin(pixmap_moved)
            painter.drawPixmap((pixmap_moved.width() - pixmap_rotated.width()) / 2.0, (pixmap_moved.height() - pixmap_rotated.height()) / 2.0, pixmap_rotated)
            painter.end()
            self._pixmaps+=[pixmap_moved.scaled(self._width, self._height)]
        
    def setProgress(self, progress):
        if self._pixmaps!=None:
            self.setPixmap(self._pixmaps[progress])
        
    def start(self):
        self.setProgress(0)
        self._progressTimeLine.start()
        
    def stop(self):
        self._progressTimeLine.stop()
