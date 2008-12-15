from qt import *

class Tab(QSplitter):
    def __init__(self, parent,name,filename=""):
        """ constructor """
        self.parentwindow=parent.parent().parent()
        QSplitter.__init__(self,parent,name)
        parent.addTab(self,name)

        self.filename=filename
        self.objects=[]
        self.selected_object=None
        self.selected_objects=[]

        self.connect(self,PYSIGNAL("clearObjects()"),self.clearObjects)

    def clearObjects(self):
        """ Clear objects """
        if hasattr(self.parentwindow,"debug_calls"):
            self.parentwindow.calls+=["Tab.clearObjects"]
        
        self.objects=[]
        self.selected_object=None
        self.selected_objects=[]
