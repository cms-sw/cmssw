from PhysicsTools.HeppyCore.particles.met import MET as BaseMET
from ROOT import TLorentzVector

class Met(BaseMET):
    
    def __init__(self, fccmet):
        self.fccmet = fccmet
        self._sum_et = fccmet.ScalarSum()
        self._tlv = TLorentzVector()
        self._tlv.SetPtEtaPhiM(fccmet.Magnitude(), 0.,fccmet.Phi(),0. )
        self._charge = 0. 
