from Vispa.Main.AbstractTab import *
from Vispa.Main.PropertyView import *

class SplitterTab(QSplitter, AbstractTab):
    """ A Tab with a QSplitter and a function to create the PropertyView.
    """
    def __init__(self, parent=None):
        AbstractTab.__init__(self)
        QSplitter.__init__(self, parent)
        
        self.adjustSize()
        self.resize(self.childrenRect().size())
        self._propertyView = None
        
    def createPropertyView(self):
        self._propertyView = PropertyView(self, "PropertyView")
        
    def propertyView(self):
        return self._propertyView
