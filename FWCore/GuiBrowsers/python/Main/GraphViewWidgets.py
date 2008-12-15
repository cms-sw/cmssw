from qt import *

class BoxFrame(QFrame):
    """ QFrame with Object """
    def __init__(self,parent,name,fl,object,frame_color="black"):
        """ constructor """
        QFrame.__init__(self,parent,name,fl)
        self.obj=object
        self.frame_color=frame_color
        self.frame_width=1
        self.setPaletteBackgroundColor(QColor("white"))
        self.setFrameShape(QFrame.Box)
        self.setFrameShadow(QFrame.Plain)
        self.setLineWidth(self.parent().parent()._lineWidth)

    def drawFrame(self,p):
        """ draw frame """
        linewidth=self.lineWidth()*self.frame_width
        p.setPen(QPen(QColor(self.frame_color),linewidth))
        p.drawRect(linewidth/2,linewidth/2,self.width()-linewidth/2,self.height()-linewidth/2)

    def mousePressEvent(self,event):
        """ When box is clicked send signal to Window """
        QFrame.mousePressEvent(self,event)
        self.parent().parent().parentwindow.emit(PYSIGNAL("selectObject()"),(self,True))

    def getMothersInObjects(self,objects):
        """ get all mothers of an object in a list of objects """
        mothers=[]
        if hasattr(self,"getMothers"):
            for o in self.getMothers():
                if hasattr(o,"graphicObject") and o.graphicObject!=[]:
                    if o.graphicObject[0] in objects or o in objects: 
                        mothers+=[o.graphicObject[0]]
        return mothers

    def __getattr__(self,name):
        return getattr(self.obj,name)

class LineWidget(QWidget):
    """ QWidget for drawing line """
    colorloop=0
    colors=["black","red","green","blue","cyan","magenta"]
    def __init__(self,parent,name,connection,col=0,connectionType=0):
        """ constructor """
        self.conn=connection
        QWidget.__init__(self,parent,name)
        self._orientation=False
        self._connectionType=connectionType
        self.setAutoMask(True)
        self.color=col
        self.setPaletteForegroundColor(QColor(LineWidget.colors[col]))
        self.setPaletteBackgroundColor(QColor(LineWidget.colors[col]))
    
    def updateMask(self):
        """ Make Widget transparent except for a line """
        bm=QBitmap( self.size() )
        bm.fill( bm.color0 )
        paint=QPainter();
        paint.begin( bm, self )
        pen=QPen( bm.color1, self.parent().parent()._connectionWidth )
        paint.setPen( pen )
        if self._connectionType==0:
         if self._orientation:
            if self.width()>self.height():
                horizontal=int(self.height()/2)
                paint.drawLine(self.width()-0,0,self.width()-0-horizontal,0+horizontal)
                paint.drawLine(self.width()-0-horizontal,0+horizontal,0+horizontal,self.height()-0-horizontal)
                paint.drawLine(0,self.height()-0,0+horizontal,self.height()-0-horizontal)
            else:
                vertical=int(self.width()/2)
                paint.drawLine(self.width()-0,0,self.width()-0-vertical,0+vertical)
                paint.drawLine(self.width()-0-vertical,0+vertical,0+vertical,self.height()-0-vertical)
                paint.drawLine(0,self.height()-0,0+vertical,self.height()-0-vertical)
         else:
            if self.width()>self.height():
                horizontal=int(self.height()/2)
                paint.drawLine(self.width()-0,self.height()-0,self.width()-0-horizontal,self.height()-0-horizontal)
                paint.drawLine(self.width()-0-horizontal,self.height()-0-horizontal,0+horizontal,0+horizontal)
                paint.drawLine(0,0,0+horizontal,0+horizontal)
            else:
                vertical=int(self.width()/2)
                paint.drawLine(self.width()-0,self.height()-0,self.width()-0-vertical,self.height()-0-vertical)
                paint.drawLine(self.width()-0-vertical,self.height()-0-vertical,0+vertical,0+vertical)
                paint.drawLine(0,0,0+vertical,0+vertical)
        else:
         if self._orientation:
                paint.drawLine(0,self.height()-1,self.width()-1,0)
         else:
                paint.drawLine(0,0,self.width()-1,self.height()-1)
        paint.end()
        self.setMask( bm )
        self.mask=bm

    def __getattr__(self,name):
        return getattr(self.conn,name)

def drawConnections(gv,this_frame,connections,connectionType=0):
        """ Draw connections of a box """
        new_connections=[]
        for plug in this_frame.sinks:
                if plug.connection.sinkQFrame!=None:
                    col=-1
                    for l in connections:
                        if l.source==plug.connection.source:
                            col=l.color
                    if col<0:
                        for l in connections:
                            if l.sink==plug.connection.source:
                                col=l.color
                    if col<0:
                        LineWidget.colorloop+=1
                        if LineWidget.colorloop>=len(LineWidget.colors):
                            LineWidget.colorloop-=len(LineWidget.colors)
                        col=LineWidget.colorloop
                    line = LineWidget(gv.viewport(),"line",plug.connection,col,connectionType)
                    gv.addChild(line)
                    if plug.label=="":
                        left=plug.connection.sinkQFrame.x()
                        top=plug.connection.sinkQFrame.y()+int(plug.connection.sinkQFrame.height()/2.0)
                        right=plug.connection.sourceQFrame.x()+plug.connection.sourceQFrame.width()
                        bottom=plug.connection.sourceQFrame.y()+plug.connection.sourceQFrame.height()
                    else:
                        left=plug.connection.sinkQLabel.x()+plug.connection.sinkQFrame.x()+int(gv._fontSize/2.0)
                        top=plug.connection.sinkQLabel.y()+plug.connection.sinkQFrame.y()+int(gv._fontSize/2.0)
                        right=plug.connection.sourceQFrame.x()+plug.connection.sourceQFrame.width()
                        bottom=plug.connection.sourceQFrame.y()+plug.connection.sourceQFrame.height()
                    line._orientation=False
                    if top>bottom:
                        top,bottom=bottom,top
                        line._orientation=not line._orientation
                    if left>right:
                        left,right=right,left
                        line._orientation=not line._orientation
                    line.setGeometry(QRect(left,top,right-left,bottom-top))
                    line.show()
                    new_connections+=[line]
        connections+=new_connections
        return new_connections

def drawLabel(gv,this_frame):
        """ Draw label in frame """
        FrameBoxLayout = QVBoxLayout(this_frame.FrameLayout,gv._boxMargin,this_frame.label+"_boxlayout")

        for property,name,value in gv._box_properties:
            if getattr(gv,property+"OnOffAction").isOn():
                if hasattr(this_frame,value):
                    prop=getattr(this_frame,value)
                    if callable(prop):
                        prop=prop()
                    if isinstance(prop,float):
                        prop="%g" % prop
                    prop=str(prop)
                    if prop!="":
                        if name!="":
                            prop=name+": "+prop
                        label = QLabel(this_frame,prop+"_label")
                        label.setText(prop)
                        FrameBoxLayout.addWidget(label)

        FrameBoxLayout.addStretch()

def updateMothersSize(gv,object,objects):
            """ update size of mother objects for box in box layout """
            child=object
            mothers=object.getMothersInObjects(objects)
            for mother in mothers:
                if child.x()-mother.x()+child.width()+gv._boxSpacing>mother.width():
                    mother.resize(child.x()-mother.x()+child.width()+gv._boxSpacing,mother.height());
                if child.y()+child.height()-mother.y()+gv._boxSpacing>mother.height():
                    mother.resize(mother.width(),child.y()+child.height()-mother.y()+gv._boxSpacing);
                child=mother

def drawSinkLabels(gv,this_frame,objects):
        FrameInputLayout = QVBoxLayout(this_frame.FrameLayout,gv._boxMargin,this_frame.label+"_inputlayout")

        show_inputtags=False
        configobjects=[]
        for o in objects:
                configobjects+=[o.obj]
        for plug in this_frame.sinks:
                plug.connection.sinkQFrame=None
                if plug.connection.source in configobjects:
                    plug.connection.sinkQFrame=this_frame
                    if plug.label!="":
                        show_inputtags=True
                        label = QLabel(this_frame,plug.label+"_label")
                        label.setText("O "+plug.label)
                        FrameInputLayout.addWidget(label)
                        plug.connection.sinkQLabel=label
        for plug in this_frame.sources:
                plug.connection.sourceQFrame=this_frame

        FrameInputLayout.addStretch()

        return show_inputtags

def drawBox(gv,object,objects,frame_left,frame_top,boxInBox=True,drawConnections=False):
        """ Draw box at posision """
        this_frame = BoxFrame(gv.viewport(),object.label+"_frame",0,object)
        if not hasattr(object,"graphicObject"):
            object.graphicObject=[]
        object.graphicObject.insert(0,this_frame)
        gv.addChild(this_frame)
        
        this_frame.FrameLayout = QHBoxLayout(this_frame,gv._boxMargin,gv._boxMargin,this_frame.label+"_layout")

        if drawConnections and drawSinkLabels(gv,this_frame,objects):
                FrameLineLayout = QVBoxLayout(this_frame.FrameLayout,gv._boxMargin,this_frame.label+"_linelayout")
                line = QFrame(this_frame,"line")
                line.setFrameShape(QFrame.VLine)
                line.setFrameShadow(QFrame.Plain)
                line.setLineWidth(gv._lineWidth)
                FrameLineLayout.addWidget(line)

        drawLabel(gv,this_frame)

        this_frame.FrameLayout.addStretch()
        this_frame.setGeometry(QRect(frame_left,frame_top,this_frame.FrameLayout.minimumSize().width(),this_frame.FrameLayout.minimumSize().height()))
        this_frame.own_height=this_frame.height()
        this_frame.own_width=this_frame.width()
        this_frame.show()
        objects+=[this_frame]

        if boxInBox:
            updateMothersSize(gv,this_frame,objects)
        
        return this_frame
