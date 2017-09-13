from PhysicsTools.HeppyCore.particles.jet import Jet as BaseJet
from rootobj import RootObj

class Jet(BaseJet, RootObj):
    def __init__(self, tlv):
        super(Jet, self).__init__()
        self._tlv = tlv

        
