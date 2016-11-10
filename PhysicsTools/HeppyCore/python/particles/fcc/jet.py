from PhysicsTools.HeppyCore.particles.jet import Jet as BaseJet
from pod import POD

from ROOT import TLorentzVector
import math

class Jet(BaseJet, POD):
    
    def __init__(self, fccobj):
        super(Jet, self).__init__(fccobj)
        self._tlv = TLorentzVector()
        p4 = fccobj.Core().P4
        self._tlv.SetXYZM(p4.Px, p4.Py, p4.Pz, p4.Mass)
        

