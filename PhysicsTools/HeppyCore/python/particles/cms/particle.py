from PhysicsTools.HeppyCore.particles.particle import Particle as BaseParticle
from ROOT import TLorentzVector

class Particle(BaseParticle):
    def __init__(self, candidate):
        self.candidate = candidate
        self._charge = candidate.charge()
        self._pid = candidate.pdgId()
        self._status = candidate.status()
        self._tlv = TLorentzVector()
        p4 = candidate.p4()
        self._tlv.SetPtEtaPhiM(p4.pt(), p4.eta(), p4.phi(), p4.mass())
        
    def __getattr__(self, attr): 
        return getattr(self.candidate, attr)
