class Zoomable(object):
    """ Interface for zoomable objects.
    """
    def __init__(self):
        #self._zoom = 1.0    # not just set self._zoom
        self.setZoom(100)    # call setZoom() because it might be overwritten
    
    def setZoom(self, zoom):
        """Takes zoom factor in percent.
        """
        self._zoom = 0.01 * zoom
        
    def zoom(self):
        """Returns zoom factor in percent.
        """
        return self._zoom * 100.0
    
    def zoomFactor(self):
        return self._zoom
    
    def incrementZoom(self):
        """Increment zoom by 10 %
        """
        self.setZoom(self._zoom * 110)
    
    def decrementZoom(self):
        """Decrement zome by 10 %
        """
        self.setZoom(self._zoom * 90)
        