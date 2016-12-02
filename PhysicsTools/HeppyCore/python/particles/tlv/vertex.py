from PhysicsTools.HeppyCore.particles.vertex import Vertex as BaseVertex
from rootobj import RootObj

import math

class Vertex(BaseVertex, RootObj):
    def __init__(self, vector3, ctau=0):
        super(Vertex, self).__init__()
        self.incoming = []
        self.outgoing = []
        self._point = vector3
        self._ctau = ctau
