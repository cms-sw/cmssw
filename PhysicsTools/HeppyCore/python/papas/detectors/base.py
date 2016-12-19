import operator

class Material(object):
    def __init__(self, name, x0, lambdaI):
        self.name = name
        self.x0 = x0
        self.lambdaI = lambdaI

        
material_CMS_ECAL = Material('CMS_ECAL', 8.9e-3, 0.25)
material_CMS_HCAL = Material('CMS_HCAL', None, 0.17)
material_void = Material('void', 0., 0.)


class DetectorElement(object):
    def __init__(self, name, volume, material, field, cluster_size=0.1):
        self.name = name
        self.volume = volume
        self.material = material
        self.field = field
        self.cluster_size = cluster_size

        
class Detector(object):
    #TODO validate geometry consistency (no hole, no overlapping volumes)
    def __init__(self):
        self.elements = dict()
        self._cylinders = []
        
    def cylinders(self):
        if len(self._cylinders):
            return self._cylinders
        for element in self.elements.values():
            if element.volume.inner is not None: 
                self._cylinders.append(element.volume.inner)
            self._cylinders.append(element.volume.outer)
        self._cylinders.sort(key=operator.attrgetter("rad"))
        return self._cylinders
