from PhysicsTools.HeppyCore.particles.particle import Particle as BaseParticle
from rootobj import RootObj
from ROOT import TVector3
from vertex import Vertex 

import math

class Particle(BaseParticle, RootObj):
    def __init__(self, pdgid, charge, tlv, status=1):
        super(Particle, self).__init__()
        self._pid = pdgid
        self._charge = charge
        self._tlv = tlv
        self._status = status
        self._start_vertex = Vertex(TVector3(),0)
        self._end_vertex = None
