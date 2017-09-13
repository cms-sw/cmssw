import operator

class DetectorElement(object):

    def __init__(self, name, volume, material):
        self.name = name
        self.volume = volume
        self.material = material
    
class Detector(object):
    #TODO validate geometry consistency (no hole, no overlapping volumes)
    def __init__(self):
        self.elements = dict()
        self._cylinders = []
        
    def cylinders(self):
        '''Return list of surface cylinders sorted by increasing radius.'''
        if len(self._cylinders):
            return self._cylinders
        for element in self.elements.values():
            if element.volume.inner is not None: 
                self._cylinders.append(element.volume.inner)
            self._cylinders.append(element.volume.outer)
        self._cylinders.sort(key=operator.attrgetter("rad"))
        return self._cylinders
