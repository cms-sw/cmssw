class Zoomable(object):
    """ Interface for zoomable objects.
    """
    def __init__(self):
        self._zoomFactor = 1.0   # not just set self._zoomFactor
        self.setZoom(100)        # call setZoom() because it might be overwritten
    
    def setZoom(self, zoom):
        """Takes zoom factor in percent.
        """
        # prevent division by zero
        self._zoomFactor = 0.01 * max(abs(zoom), 0.000001)

    def zoom(self):
        """Returns zoom factor in percent.
        """
        return self._zoomFactor * 100.0
    
    def zoomFactor(self):
        return self._zoomFactor 
    
    def incrementZoom(self):
        """Increment zoom by 10 %
        """
        self.setZoom(self._zoomFactor * 110)
    
    def decrementZoom(self):
        """Decrement zome by 10 %
        """
        self.setZoom(self._zoomFactor * 90)
        