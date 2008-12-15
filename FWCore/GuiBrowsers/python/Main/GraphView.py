import os.path

from Icons import *
from GraphViewWidgets import *

from qt import *

class GraphView(QScrollView):
    """ QScrollView that holds the BoxFrames """
    def __init__(self, parent=None, name=None, fl=0):
        """ constructor """
        self.parentwindow=parent.parent().parent().parent().parent()

        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.__init__"]
        
        QScrollView.__init__(self,parent,name,fl)
        if not hasattr(self,"_box_properties"):
            self._box_properties=[]
#        self.enableClipper(True);
        self.parenttab=parent
        self.viewport().setPaletteForegroundColor(QColor("black"))
        self.viewport().setPaletteBackgroundColor(QColor("white"))
        self.fillMenu()
        self.setZoomFactor(100)
        self.readIni()
        self._objects=[]
        self._selected_object=None
        self._connections=[]
        self._extra_objects=[]
        self._extra_connections=[]
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding,0)
        LineWidget.colorloop=0
        
        self.connect(self.parenttab,PYSIGNAL("clearObjects()"),self.clearObjects)
        self.connect(self.parenttab,PYSIGNAL("tabVisible()"),self.showMenu)
        self.connect(self.parenttab,PYSIGNAL("tabInvisible()"),self.hideMenu)

    def fillMenu(self):
        """ fill entries in MainMenu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.fillMenu"]
        
        self.GraphView = QPopupMenu(self.parentwindow)
        self.GraphView.id=self.parentwindow.MenuBar.count()+1
        self.parentwindow.MenuBar.insertItem(QString(""),self.GraphView,self.GraphView.id)
        self.parentwindow.MenuBar.findItem(self.GraphView.id).setText("&GraphView")

        self.zoomAction = QAction("&Zoom...","Ctrl+Z",self.parentwindow,"zoomAction")
        self.connect(self.zoomAction,SIGNAL("activated()"),self.zoomDialog)

        self.image0 = QPixmap()
        self.image0.loadFromData(save_image,"PNG")
        self.screenshotAction = QAction(QIconSet(self.image0),"&Save screenshot...","Ctrl+S",self.parentwindow,"screenshotAction")
        self.connect(self.screenshotAction,SIGNAL("activated()"),self.screenshot)

        self.zoomAction.addTo(self.GraphView)
        self.screenshotAction.addTo(self.GraphView)

        self.toolBar=QToolBar(QString("GraphView"),self.parentwindow,Qt.DockTop)
        self.zoomEdit=QLineEdit(QString("100"),self.toolBar)
        self.zoomEdit.setValidator(QIntValidator(1,1000,self))
        self.zoomEdit.setFixedWidth(30)
        self.connect(self.zoomEdit,SIGNAL("returnPressed()"),self.zoomChanged)
        self.zoomLabel=QLabel(QString("%"),self.toolBar)
        self.screenshotAction.addTo(self.toolBar)

        self.GraphView.insertSeparator()
        for property,name,value in self._box_properties:
            setattr(self,property+"OnOffAction",QAction("&"+property+" On/Off","",self.parentwindow,property+"OnOffAction"))
            getattr(self,property+"OnOffAction").setToggleAction(1)
            getattr(self,property+"OnOffAction").setOn(value=="label" or value=="getShortLabelWithType")
            getattr(self,property+"OnOffAction").addTo(self.GraphView)
            self.connect(getattr(self,property+"OnOffAction"),SIGNAL("activated()"),self.viewChanged)
            
    def showMenu(self):
        """ show entries in MainMenu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.showMenu"]
        
        self.parentwindow.MenuBar.setItemVisible(self.GraphView.id,True)
        self.zoomAction.setVisible(True)
        self.screenshotAction.setVisible(True)
        self.zoomEdit.show()
        self.zoomLabel.show()
        self.toolBar.show()
        
        for property,name,value in self._box_properties:
            getattr(self,property+"OnOffAction").setVisible(True)
        
    def hideMenu(self):
        """ hide entries in MainMenu """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.hideMenu"]
        
        self.parentwindow.MenuBar.setItemVisible(self.GraphView.id,False)
        self.zoomAction.setVisible(False)
        self.screenshotAction.setVisible(False)
        self.zoomEdit.hide()
        self.zoomLabel.hide()
        self.toolBar.hide()
        
        for property,name,value in self._box_properties:
            getattr(self,property+"OnOffAction").setVisible(False)
       
    def allObjects(self):
        """ return all objects in GraphView """
        all_objects=self._objects[:]
        for eo in self._extra_objects:
            all_objects+=eo
        return all_objects

    def updateSize(self):
        """ resize view according to contents """
        width=0
        height=0
        for object in self.allObjects():
            if width<object.x()+self.contentsX()+object.width()+self._boxSpacing:
                width=object.x()+self.contentsX()+object.width()+self._boxSpacing
            if height<object.y()+self.contentsY()+object.height()+self._boxSpacing:
                height=object.y()+self.contentsY()+object.height()+self._boxSpacing
        self.resizeContents(width,height)

    def deleteObjects(self,objects):
        """ Delete objects from GraphView """
        for entry in objects:
            if hasattr(entry,"obj") and hasattr(entry.obj,"graphicObject"):
                entry.obj.graphicObject=entry.obj.graphicObject[1:]
            if hasattr(entry,"obj"):
                del entry.obj
            entry.parent().removeChild(entry)
            entry.close()
            del entry

    def clearObjects(self):
        """ Clear objects in GraphView """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.clearObjects"]

        self.deleteObjects(self._objects)
        self.deleteObjects(self._connections)
        self._objects=[]
        self._connections=[]
        for eo in self._extra_objects:
            index=self._extra_objects.index(eo)
            self.deleteObjects(self._extra_objects[index])
            self.deleteObjects(self._extra_connections[index])
        self._extra_objects=[]
        self._extra_connections=[]
        self.setContentsPos(0,0)
        self.updateSize()
        LineWidget.colorloop=0

    def viewChanged(self):
        """ Show list of Objects in GraphView """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.viewChanged"]
        
        self.writeIni()
        self.parentwindow.saveWindowState()
        self.parentwindow.restoreWindowState()
        
    def setZoomFactor(self,zoom):
        """ set Zoom factor of GraphView """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.setZoomFactor"]
        
        self._zoom=zoom
        self.zoomEdit.setText(str(int(self._zoom)))
        self._boxMargin=max(1,int(6.0*self._zoom/100.0))
        self._childrenIndent=max(1,int(14.0*self._zoom/100.0))
        self._boxSpacing=max(1,int(8.0*self._zoom/100.0))
        self._lineWidth=max(1,int(1.5*self._zoom/100.0))
        self._connectionWidth=max(1,int(2.5*self._zoom/100.0))
        self._fontSize=max(1,int(14.0*self._zoom/100.0))
        font=self.font()
        font.setPixelSize(self._fontSize)
        self.setFont(font)

    def zoomDialog(self):
        """ Show zoom dialog """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.zoomDialog"]
        
        zoom = QInputDialog.getInteger("Zoom...","Enter zoom level in %:",self._zoom,1,1000)[0]
        self.zoomEdit.setText(str(zoom))
        self.zoomChanged()
                
    def zoomChanged(self):
        """ update zoom """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.zoomChanged"]

        if int(self.zoomEdit.text())!=int(self._zoom):
            self.setZoomFactor(int(self.zoomEdit.text()))
            self.viewChanged()
            
    def grabChildWidgets(self,widget,w,h):
        """ grabChildWidgets function that allows transparent widgets """
        res=QPixmap( w,h )
        res.fill( widget, QPoint( 0, 0 ) )
        wr=QRect( 0, 0, w, h )
        if ( res.isNull() and w ):
            return res
        if hasattr(widget,"mask"):
            res.setMask(widget.mask)
        QPainter.redirect( widget, res )
        e=QPaintEvent( wr, False )
        QApplication.sendEvent( widget, e )
        QPainter.redirect( widget, widget )
        children = widget.children()
        if ( children!=None ):
            p=QPainter( res )
            for child in children:
                if ( child.isWidgetType() and
                    not child.isHidden() and
                    not child.isTopLevel() and
                    child.geometry().intersects(wr) ):
                    childpos = child.pos()
                    cpm = self.grabChildWidgets( child,child.width(),child.height() )
                    if ( cpm.isNull() ):
                        res.resize( 0, 0 )
                        break
                    p.drawPixmap( childpos, cpm)
        return res

    def grabWidget(self,widget,x=0,y=0,w=-1,h=-1):
        """ grabWidget function that allows transparent widgets """
        res=QPixmap()
        if ( widget==None ):
            return res
        if ( w < 0 ):
            w = widget.width() - x
        if ( h < 0 ):
            h = widget.height() - y
        wr=QRect( x, y, w, h )
        if ( wr == widget.rect() ):
            return self.grabChildWidgets( widget,w,h )
        if ( not wr.intersects( wr ) ):
            return res
        res.resize( w, h )
        if( res.isNull() ):
            return res;
        res.fill( widget, QPoint( w,h ) )
        tmp=self.grabChildWidgets( widget,w,h )
        if( tmp.isNull() ):
            return tmp
        bitBlt( res, 0, 0, tmp, x, y, w, h )
        return res

    def screenshot(self,fileName=None):
        """ save screenshot """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.screenshot"]
        
        filter=QString("")
        while fileName==None or (os.path.exists(fileName) and not QMessageBox.question(self,"Overwrite file","File already exists. Do you want to overwrite?",QMessageBox.Yes,QMessageBox.Cancel)==QMessageBox.Yes):
            fileName = str(QFileDialog.getSaveFileName(".","BMP Image (*.bmp);;PNG Image (*.png)",self,"Save image dialog","Save image...",filter))
        if fileName!="":
            self.parentwindow.StatusBar.message("saving screenshot...")
            name=fileName
            ext=str(filter).split(" ")[0]
            if os.path.splitext(fileName)[1].upper().strip(".") in ["BMP","PNG"]:
                name=os.path.splitext(fileName)[0]
                ext=os.path.splitext(fileName)[1].upper().strip(".")
            x=self.contentsX()
            y=self.contentsY()
            w=self.contentsWidth()
            h=self.contentsHeight()
            self.setContentsPos(0,0)
            picture=self.grabWidget(self.viewport(),0,0,w,h)
            self.setContentsPos(x,y)
            picture.save(name+"."+ext.lower(),ext)
            self.parentwindow.StatusBar.message("saving screenshot...done")
        
    def mousePressEvent(self,event):
        """ When box is clicked send signal to Window """
        QScrollView.mousePressEvent(self,event)
        if not self.viewport().childrenRegion().contains(event.pos()):
            self.parentwindow.emit(PYSIGNAL("selectObject()"),(None,))

    def wheelEvent(self,event):
        """ Scroll when mousewheel turned """
        if event.state()&QMouseEvent.ControlButton:
            self.zoomEdit.setText(str(int(self._zoom*(1.0+event.delta()/120.0/10.0))))
            self.zoomChanged()
        else:
            QScrollView.wheelEvent(self,event)

    def readIni(self):
        """ read options from ini """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.readIni"]
        
        ini=self.parentwindow.loadIni()
        if ini.has_option("GraphView", "zoom"):
            self.setZoomFactor(ini.getint("GraphView", "zoom"))
        for property,name,value in self._box_properties:
            if ini.has_option("GraphView",property):
                if hasattr(self,property+"OnOffAction"):
                    getattr(self,property+"OnOffAction").setOn(ini.getboolean("GraphView",property))

    def writeIni(self):
        """ write options to ini """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["GraphView.writeIni"]
        
        ini=self.parentwindow.loadIni()
        if not ini.has_section("GraphView"):
            ini.add_section("GraphView")
        ini.set("GraphView","zoom",self._zoom)
        for property,name,value in self._box_properties:
            if hasattr(self,property+"OnOffAction"):
                ini.set("GraphView",property,getattr(self,property+"OnOffAction").isOn())
        self.parentwindow.saveIni(ini)
