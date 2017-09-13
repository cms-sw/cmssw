from PhysicsTools.HeppyCore.particles.vertex import Vertex as BaseVertex
from pod import POD

from ROOT import TVector3

class Vertex(BaseVertex, POD):

    def __init__(self, fccobj):
        super(Vertex, self).__init__(fccobj)
        self.incoming = []
        self.outgoing = []
        self._point = TVector3(fccobj.Position().X,
                               fccobj.Position().Y,
                               fccobj.Position().Z)
        self._point *= 1e-3 # pythia : mm -> papas : m
        self._ctau = fccobj.Ctau()
        
