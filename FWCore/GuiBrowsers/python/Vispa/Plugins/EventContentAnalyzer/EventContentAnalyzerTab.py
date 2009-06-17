import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Main.SplitterTab import *
from EventContentView import *

class EventContentAnalyzerTab(SplitterTab):
    """
    """
    def __init__(self, parent=None):
        logging.debug(__name__ + ": __init__")
        SplitterTab.__init__(self, parent)

        self._centerView = None
        self._createCenterView()
        
    def _createCenterView(self):
        self._centerView = EventContentView(self)
        
    def centerView(self):
        return self._centerView
