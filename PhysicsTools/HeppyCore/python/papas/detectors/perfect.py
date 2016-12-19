from detector import Detector, DetectorElement
import material
from geometry import VolumeCylinder
import math

class ECAL(DetectorElement):

    def __init__(self):
        volume = VolumeCylinder('ecal', 1.55, 2.25, 1.30, 2. )
        mat = material.Material('ECAL', 8.9e-3, 0.) # lambda_I = 0
        self.eta_crack = 1.5
        self.emin = 2.
        super(ECAL, self).__init__('ecal', volume,  mat)

    def energy_resolution(self, energy, theta=0.):
        return 0.

    def energy_response(self, energy, eta):
        return 1.

    def cluster_size(self, ptc):
        pdgid = abs(ptc.pdgid())
        if pdgid==22 or pdgid==11:
            return 0.04
        else:
            return 0.07

    def acceptance(self, cluster):
        return True

    def space_resolution(self, ptc):
        pass

    
class HCAL(DetectorElement):

    def __init__(self):
        volume = VolumeCylinder('hcal', 2.9, 3.6, 1.9, 2.6 )
        mat = material.Material('HCAL', None, 0.17)
        super(HCAL, self).__init__('ecal', volume, mat)

    def energy_resolution(self, energy, theta=0.):
        return 0.

    def energy_response(self, energy, eta):
        return 1.

    def cluster_size(self, ptc):
        return 0.2

    def acceptance(self, cluster):
        return True
    
    def space_resolution(self, ptc):
        pass


    
class Tracker(DetectorElement):
    #TODO acceptance and resolution depend on the particle type
    
    def __init__(self):
        volume = VolumeCylinder('tracker', 1.29, 1.99)
        mat = material.void
        super(Tracker, self).__init__('tracker', volume,  mat)

    def acceptance(self, track):
        return True

    def pt_resolution(self, track):
       return 0.

    

class Field(DetectorElement):

    def __init__(self, magnitude):
        self.magnitude = magnitude
        volume = VolumeCylinder('field', 2.9, 3.6)
        mat = material.void
        super(Field, self).__init__('tracker', volume,  mat)
        
        
class Perfect(Detector):
    '''A detector with the geometry of CMS and the same cluster size, 
    but without smearing, and with full acceptance (no thresholds).
    Used for testing purposes. 
    '''
    def __init__(self):
        super(Perfect, self).__init__()
        self.elements['tracker'] = Tracker()
        self.elements['ecal'] = ECAL()
        self.elements['hcal'] = HCAL()
        self.elements['field'] = Field(3.8)

perfect = Perfect()
