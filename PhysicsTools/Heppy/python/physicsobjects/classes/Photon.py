from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *

class Photon(PhysicsObject ):

    '''                                                                                                                                                                                                                                                                return object from the photon 
    '''
    def hOVERe(self):
#        return self.physObj.full5x5_hadTowOverEm()
        hadTowDepth1O = self.physObj.hadTowDepth1OverEm() * (self.physObj.superCluster().energy()/self.physObj.full5x5_e5x5() if self.physObj.full5x5_e5x5() else 1)
        hadTowDepth2O = self.physObj.hadTowDepth2OverEm() * (self.physObj.superCluster().energy()/self.physObj.full5x5_e5x5() if self.physObj.full5x5_e5x5() else 1)
        return hadTowDepth1O + hadTowDepth2O

    def r9(self):
        return self.physObj.full5x5_r9()

    def sigmaIetaIeta(self):
        return self.physObj.full5x5_sigmaIetaIeta()

    def chargedHadronIso(self):
        return self.physObj.chargedHadronIso()

    def neutralHadronIso(self):
        return self.physObj.neutralHadronIso()

    def photonIso(self):
        return self.physObj.photonIso()

    pass
