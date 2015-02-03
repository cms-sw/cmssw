from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *

#add __str__ to reco::GenParticle python wrapper
import ROOT
def printGenParticle(self):
        tmp = '{className} : {pdgId:>3}, pt = {pt:5.1f}, eta = {eta:5.2f}, phi = {phi:5.2f}, mass = {mass:5.2f}'
        base= tmp.format( className = self.__class__.__name__,
                           pdgId = self.pdgId(),
                           pt = self.pt(),
                           eta = self.eta(),
                           phi = self.phi(),
                           mass = self.mass() )
        theStr = '{base}, status = {status:>2}'.format(base=base, status=self.status())
        return theStr
setattr(ROOT.reco.GenParticle,"__str__",printGenParticle)

#from ROOT.reco import GenParticle   # sometimes doesn't work
GenParticle = ROOT.reco.GenParticle  # this instead does
