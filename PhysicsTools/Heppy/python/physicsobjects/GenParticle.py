from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *

#add __str__ to reco::GenParticle python wrapper
import ROOT
def printGenParticle(self):
        base = basePrint(self)
        theStr = '{base}, status = {status:>2}'.format(base=base, status=self.status())
        return theStr
setattr(ROOT.reco.GenParticle,"basePrint",Particle.__str__)
setattr(ROOT.reco.GenParticle,"__str__",printGenParticle)

from ROOT.reco import GenParticle
