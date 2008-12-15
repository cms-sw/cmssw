from qt import *
from qttable import *

class PropertiesTextEdit(QTextEdit):
    """ QTextEdit for use in QTable """
    def __init__(self, parent=None,text=""):
        """ constructor """
        QTextEdit.__init__(self,parent)
        self.setWordWrap(QTextEdit.WidgetWidth)
        self.setFrameStyle( QFrame.NoFrame )
        self.setReadOnly(True)
        self.setText(text)
            
    def sizeHint(self):
        """ Calculate correct size for QTable """
        width=self.contentsWidth()+3
        height=self.contentsHeight()+3
        if self.verticalScrollBar().isVisible():
            width+=self.verticalScrollBar().width()
        if self.horizontalScrollBar().isVisible():
            height+=self.horizontalScrollBar().height()
        return QSize(width,height)

    def wheelEvent(self,event):
        """ Scroll when mousewheel turned """
        self.parent().parent().parent().wheelEvent(event)

class PropertiesView(QTable):
    """ QTable to show Properties """
    def __init__(self, parent=None,name=None):
        """ Constructor """
        self.parentwindow=parent.parent()
        
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["PropertiesView.__init__"]
        
        QTable.__init__(self,parent,name)
        self.updateIni=False
        self.setSorting(False)
        self.setLeftMargin(0)
        self.verticalHeader().hide()
        self.setNumCols(2)
        self.horizontalHeader().setLabel(0,"Property")
        self.horizontalHeader().setLabel(1,"Value")
        self.readIni()
        self.setReadOnly(True)
        self.setSelectionMode(QTable.NoSelection)
        self.setFocusStyle(QTable.FollowStyle)
        self.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Expanding,0)
        self._rows=0

        self._object=None
        
        self.connect(self.parentwindow,PYSIGNAL("clearObjects()"),self.clearObjects)
        self.connect(self.parentwindow,PYSIGNAL("objectSelected()"),self.showObject)
    
    def setRows(self,num=0):
        """ Set number of rows """
        self._rows=num
        self.setNumRows(self._rows)

    def addRow(self,w1,w2):
        """ Add row with two widgets """
        self._rows+=1
        self.setNumRows(self._rows)
        self.setCellWidget(self._rows-1,0,w1)
        self.setCellWidget(self._rows-1,1,w2)
        self.adjustRow(self._rows-1)

    def addLabel(self,name="",value=""):
        """ Add row with two Labels """
        self.addRow(QLabel(name,self),QLabel(value,self))

    def addTextEdit(self,name="",value=""):
        """ Add row with two TextEdits """
        self.addRow(PropertiesTextEdit(self,name),PropertiesTextEdit(self,value))

    def sizeHint(self):
        """ Calculate correct size for QDockWindow """
        width=self.contentsWidth()+3
        height=self.contentsHeight()+3
        if self.verticalScrollBar().isVisible():
            width+=self.verticalScrollBar().width()
        if self.horizontalScrollBar().isVisible():
            height+=self.horizontalScrollBar().height()
        return QSize(width,height)

    def clearObjects(self):
        """ Clear Object """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["PropertiesView.clearObjects"]
        
        self._object=None
        self.setRows(0)

    def showObject(self):
        """ Fill properties from Object """
        self.clearObjects()

        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["PropertiesView.showObjects"]
        
        self._object=self.parentwindow.selected_object
        if self._object!=None:
            for prop in self._object.properties:
                if prop[0]=="Label":
                    self.addLabel(prop[1],prop[2])
                if prop[0]=="Text":
                    self.addTextEdit(prop[1],prop[2])
        self.resizeEvent(None)

    def resizeEvent(self,event):
        """ resize columns when table size is changed """
        if event!=None:
            QTable.resizeEvent(self,event)
        space=self.width()-3
        if self.verticalScrollBar().isVisible():
            space-=self.verticalScrollBar().width()
        space-=self.columnWidth(0)
        self.setColumnWidth(1,space)
        if self.updateIni:
            self.writeIni()

    def columnWidthChanged(self,col):
        """ resize columns when column size is changed """
        QTable.columnWidthChanged(self,col)
        if col==1:
            space=self.width()-3
            if self.verticalScrollBar().isVisible():
                space-=self.verticalScrollBar().width()
            space-=self.columnWidth(1)
            if space!=self.columnWidth(0):
                self.setColumnWidth(0,space)
        else:
            space=self.width()-3
            if self.verticalScrollBar().isVisible():
                space-=self.verticalScrollBar().width()
            space-=self.columnWidth(0)
            if space!=self.columnWidth(1):
                self.setColumnWidth(1,space)
                if self.updateIni:
                    self.writeIni()

    def readIni(self):
        """ read options from ini """
        if hasattr(self,"debug_calls"):
            self.calls+=["PropertiesView.readIni"]
        
        ini=self.parentwindow.loadIni()
        width=300
        if ini.has_option("propertiesview", "width"):
            width=ini.getint("propertiesview", "width")
        self.resize(width,0)
        columnwidth=120
        if ini.has_option("propertiesview", "columnwidth"):
            columnwidth=ini.getint("propertiesview", "columnwidth")
        self.setColumnWidth(0,columnwidth)
        self.updateIni=True
        
    def writeIni(self):
        """ write options to ini """
        if hasattr(self,"debug_calls"):
            self.calls+=["PropertiesView.writeIni"]
        
        ini=self.parentwindow.loadIni()
        if not ini.has_section("propertiesview"):
            ini.add_section("propertiesview")
        ini.set("propertiesview", "width",self.width())
        ini.set("propertiesview", "columnwidth",self.columnWidth(0))
        self.parentwindow.saveIni(ini)
