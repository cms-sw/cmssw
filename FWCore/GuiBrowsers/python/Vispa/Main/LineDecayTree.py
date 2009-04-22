import logging
import sys
import random

from PyQt4 import QtGui
from PyQt4 import QtCore

from Vispa.Main.Exceptions import *
try:
    from pxl.algorithms import *
except Exception:
    logging.info(__name__ +": "+ exception_traceback())

from BasicDataAccessor import *
from RelativeDataAccessor import *
from ParticleDataAccessor import *
from Workspace import *
from PropertyView import *
from Vispa.Main.ZoomableScrollArea import *

class LineDecayTree(Workspace):
    """Visualizes a decay tree.
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        Workspace.__init__(self, parent)
        self._accessor = None
        self._dataObjects = []
        self._operationId = 0
        self.setPalette(QPalette(Qt.black, Qt.white))
        self._allNodes = {}
        self._vector = []
        self._vector2 = []  
        self._scrollArea = ZoomableScrollArea(self)
        self._selectedobject = None
        self._color = [QtGui.QColor(29,11,249), QtGui.QColor(75,240,0), QtGui.QColor(253,74,74), QtGui.QColor(247,77,251), QtGui.QColor(176,179,177), QtGui.QColor(254,244,67)]
        
                
    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the lines are created.
        
        You need to call updateContent() in order to make the changes visible.   
        """
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        if not isinstance(accessor, RelativeDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type RelativeDataAccessor.")
        if not isinstance(accessor, ParticleDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type ParticleDataAccessor.")
        self._accessor = accessor
    
    def accessor(self):
        return self._accessor
    
    def scrollArea(self):
        """ Returns scroll area of this tab.
        """
        return self._scrollArea
    
    def setDataObjects(self, objects):
        """ Sets the selected object from which the lines are created
        
        You need to call updateContent() in order to make the changes visible   
        """
        self._dataObjects = objects
        
    def dataObjects(self):
        return self._dataObjects

    def clear(self):

        logging.debug(__name__ + ": clear")
        # Abort currently ongoing drawing operations
        self._operationId += 1
        Workspace.clear(self)

    def updateContent(self):
        """ Clear the LineDecayTree and refill it.
        """
        logging.debug(__name__ + ": updateContent")
        self.clear()
        self.autolayout()
  
    def nodesfiller(self):
        for object in self._dataObjects:
            node1 = Node()
            node1.position = Vector2(0,0)
            node1.isVertex = False
            node2 = Node()
            node2.position = Vector2(0,0)
            node2.isVertex = False
            label = ""
            if hasattr(object, "getId"):
                self._allNodes[object] = [node1 , node2, label, object.getId()]
            else:
                self._allNodes[object] = [node1 , node2, label]
            
        for object in self._dataObjects:
            self._allNodes[object][2] = self._accessor.label(object)
                           
        for object1 in self._dataObjects: 
            for object2 in self._dataObjects:
                if object2 in self._accessor.daughterRelations(object1):
                    self._allNodes[object1][1] = self._allNodes[object2][0]
        
        for object1 in self._dataObjects:
            for object2 in self._dataObjects:
                if object2 in self._accessor.motherRelations(object1):               
                    self._allNodes[object1][0] = self._allNodes[object2][1]

        for object in self._dataObjects:
            self._allNodes[object][1].mothers.append(self._allNodes[object][0])            
            self._allNodes[object][0].children.append(self._allNodes[object][1])

    def autolayout(self):
        self.nodesfiller()

        vector1=[]
        vector=NodeVector()

        for object1 in self._dataObjects:
            if not self._allNodes[object1][1] in vector1:
                vector1.append(self._allNodes[object1][1])
                
        for object1 in self._dataObjects:
            if not self._allNodes[object1][0] in vector1:
                vector1.append(self._allNodes[object1][0])
                
        for i in range(len(vector1)): 
            vector.append(vector1[i])

        try:
            autolayouter=AutoLayout()
            autolayouter.init(vector)
            autolayouter.layout(True)
        except Exception:
            logging.error(__name__ +": Pxl Autolayout not found: "+ exception_traceback())
        
        miny = 1000000
        for i in range(len(vector)):
           if vector[i].position.y<miny:
               miny=vector[i].position.y

        for i in range(len(vector)):
            vector[i].position.y = vector[i].position.y - miny + 10
        
        for i in range(len(vector)):
            vector[i].position.x = vector[i].position.x - 30
             
        for object in self._dataObjects:
            if self._accessor.id(object)!=None:
                if len(self._allNodes[object][0].mothers) == 0 and len(self._allNodes[object][1].children) == 0 :
                    x1 = self._allNodes[object][0].position.x
                    y1 = self._allNodes[object][0].position.y
                    x2 = self._allNodes[object][1].position.x
                    y2 = self._allNodes[object][1].position.y      
                    label1 = self._allNodes[object][2]
#                    isVertex = self._allNodes[object][1].isVertex
                    self._vector2.append([x1,y1,x2,y2,label1,self._accessor.id(object),object])
                else:    
                    x1 = self._allNodes[object][0].position.x
                    y1 = self._allNodes[object][0].position.y
                    x2 = self._allNodes[object][1].position.x
                    y2 = self._allNodes[object][1].position.y      
                    label1 = self._allNodes[object][2]
#                   isVertex = self._allNodes[object][1].isVertex
                    self._vector.append([x1,y1,x2,y2,label1,self._accessor.id(object),object])
            else:
                if len(self._allNodes[object][0].mothers) == 0 and len(self._allNodes[object][1].children) == 0 :
                    x1 = self._allNodes[object][0].position.x
                    y1 = self._allNodes[object][0].position.y
                    x2 = self._allNodes[object][1].position.x
                    y2 = self._allNodes[object][1].position.y      
                    label1 = self._allNodes[object][2]
#                    isVertex = self._allNodes[object][1].isVertex
                    self._vector2.append([x1,y1,x2,y2,label1,0,object])
               
                else:    
                    x1 = self._allNodes[object][0].position.x
                    y1 = self._allNodes[object][0].position.y
                    x2 = self._allNodes[object][1].position.x
                    y2 = self._allNodes[object][1].position.y      
                    label1 = self._allNodes[object][2]
#                   isVertex = self._allNodes[object][1].isVertex
                    self._vector.append([x1,y1,x2,y2,label1,0,object])
            
        adjustorphans1=0
        adjustorphans2=100000
        for i in range(len(self._vector)):
           if self._vector[i][3] > adjustorphans1:
               adjustorphans1 = self._vector[i][3]
 
        for i in range(len(self._vector2)):
           if self._vector2[i][1] < adjustorphans2:
               adjustorphans2 = self._vector2[i][1]
        
        for i in range(len(self._vector2)):
            self._vector2[i][1] = self._vector2[i][1] - adjustorphans2 + adjustorphans1 + 30
            self._vector2[i][3] = self._vector2[i][3] - adjustorphans2 + adjustorphans1 + 30
    
        for object in self._dataObjects:
            for i in range(len(self._vector2)):
                if self._vector2[i][6] == object:
                    self._allNodes[object][0].position.x = self._vector2[i][0]
                    self._allNodes[object][0].position.y = self._vector2[i][1]
                    self._allNodes[object][1].position.x = self._vector2[i][2]
                    self._allNodes[object][1].position.y = self._vector2[i][3]
    
        width=0
        height=0
        
        for i in range(len(self._vector)):
            if self._vector[i][2] > width:
                width = self._vector[i][2] 
            if self._vector[i][3]  > height:
                height = self._vector[i][3]  
        for i in range(len(self._vector2)):
            if self._vector2[i][2]  > width:
                width = self._vector2[i][2]
            if self._vector2[i][3]  > height:
                height = self._vector2[i][3]    
        
        self.resize(width+20,height+10)

#    def select(self,object):
        
#        if object == self._selectedobject(object):
#            return True
#        else:
#            return False
        
    def mousePressEvent(self, event):

        zoom = self.zoomFactor()

        for object in self._dataObjects:
            k = 5 #search factor
            x1 = int((self._allNodes[object][0].position.x - k) *zoom)
            x2 = int((self._allNodes[object][1].position.x + k) *zoom)
            y1 = int(self._allNodes[object][0].position.y *zoom)
            y2 = int(self._allNodes[object][1].position.y *zoom)
            testx = range(x1,x2)
            
            if y1 - y2 < 0:
                testy = range(int(y1 - k*zoom), int(y2 + k*zoom))
               # print testy
            else:
                testy = range(int(y2 - k*zoom), int(y1 + k*zoom))
               # print testy


            if event.pos().x() in testx and event.pos().y() in testy:
#                if hasattr(object, "getId"):
#                    print object.getId()
                if object != self._selectedobject:
                    self._selectedobject = object
                    self.repaint()
                    break
                else:
                    continue
               # print object
                #test123 = Workspace()
                #test123.setDataObject(object)
               # test123.updateContent()
                
                #self.emit(SIGNAL("widgetSelected", test123.widgetByObject(object))) 
                
            else:
                self._selectedobject = None
                self.repaint()
        
    def paintEvent(self, Event):

        zoom = self.zoomFactor()
    
        
        paint = QtGui.QPainter()
        
        paint.begin(self)
        paint.setRenderHint(QPainter.Antialiasing)
        
        for object in self._dataObjects:
            x1 = self._allNodes[object][0].position.x * zoom
            y1 = self._allNodes[object][0].position.y * zoom
            x2 = self._allNodes[object][1].position.x * zoom
            y2 = self._allNodes[object][1].position.y * zoom      
            label = self._allNodes[object][2]
            
            if self._accessor.isLepton(object):
                paint.setPen(QtGui.QPen(self._color[0], 3*zoom, QtCore.Qt.SolidLine))
            
            elif self._accessor.isQuark(object):
                paint.setPen(QtGui.QPen(self._color[1], 3*zoom, QtCore.Qt.SolidLine))
                
            elif self._accessor.isBoson(object):
                paint.setPen(QtGui.QPen(self._color[2], 3*zoom, QtCore.Qt.DashLine))
                           
            elif self._accessor.isGluon(object):
                paint.setPen(QtGui.QPen(self._color[3], 3*zoom, QtCore.Qt.SolidLine))

            else:
                paint.setPen(QtGui.QPen(self._color[4], 3*zoom, QtCore.Qt.SolidLine))
            
            paint.drawLine(x1,y1,x2,y2)
            paint.setPen(QtGui.QPen(QtCore.Qt.black, 3*zoom, QtCore.Qt.SolidLine))
            paint.setFont(QtGui.QFont('Arial',10*zoom))
            paint.drawText((x2+x1)/2 ,(y2+y1)/2 - 5*zoom , label)

        for object in self._dataObjects:
            if self._selectedobject != None: 
                x1 = self._allNodes[self._selectedobject][0].position.x * zoom
                y1 = self._allNodes[self._selectedobject][0].position.y * zoom
                x2 = self._allNodes[self._selectedobject][1].position.x * zoom
                y2 = self._allNodes[self._selectedobject][1].position.y * zoom      
                label1 = self._allNodes[self._selectedobject][2]

                paint.setPen(QtGui.QPen(self._color[5], 3*zoom, QtCore.Qt.SolidLine))
                paint.drawLine(x1,y1,x2,y2)
                paint.setPen(QtGui.QPen(QtCore.Qt.black, 3*zoom, QtCore.Qt.SolidLine))
                paint.setFont(QtGui.QFont('Arial',10*zoom))
                paint.drawText((x2+x1)/2 ,(y2+y1)/2 - 5*zoom , label1)
        
        for object in self._dataObjects:
            x1 = self._allNodes[object][0].position.x * zoom
            y1 = self._allNodes[object][0].position.y * zoom
            x2 = self._allNodes[object][1].position.x * zoom
            y2 = self._allNodes[object][1].position.y * zoom      
            label1 = self._allNodes[object][2] 
            paint.setPen(QtGui.QPen(QColor(Qt.blue).lighter(140), 5*zoom, QtCore.Qt.SolidLine))
            paint.setBrush(QColor(Qt.blue).lighter(140))
            paint.drawEllipse((x1-1* zoom), (y1-1* zoom) ,2* zoom,2* zoom)
            paint.drawEllipse((x2-1* zoom), (y2-1* zoom) ,2* zoom,2* zoom)
    
        paint.end()
        
        width=0
        height=0
        
        for i in range(len(self._vector)):
            if self._vector[i][2] * zoom > width:
                width = self._vector[i][2] * zoom
            if self._vector[i][3] * zoom > height:
                height = self._vector[i][3] * zoom  
        for i in range(len(self._vector2)):
            if self._vector2[i][2] * zoom > width:
                width = self._vector2[i][2] * zoom
            if self._vector2[i][3] * zoom > height:
                height = self._vector2[i][3] * zoom    
        
        self.resize(width+20,height+10)
