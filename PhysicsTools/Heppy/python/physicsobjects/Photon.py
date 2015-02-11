from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *

class Photon(PhysicsObject ):

    '''                                                                                                                                                                                                                                                                return object from the photon 
    '''
    def hOVERe(self):
        #return self.physObj.hadronicOverEm() 
        hadTowDepth1O = self.physObj.hadTowDepth1OverEm() * (self.physObj.superCluster().energy()/self.physObj.full5x5_e5x5() if self.physObj.full5x5_e5x5() else 1)
        hadTowDepth2O = self.physObj.hadTowDepth2OverEm() * (self.physObj.superCluster().energy()/self.physObj.full5x5_e5x5() if self.physObj.full5x5_e5x5() else 1)
        return hadTowDepth1O + hadTowDepth2O

    def r9(self):
        return self.physObj.r9()

    def sigmaIetaIeta(self):
        return self.physObj.sigmaIetaIeta()

    def full5x5_r9(self):
        return self.physObj.full5x5_r9()

    def full5x5_sigmaIetaIeta(self):
        return self.physObj.full5x5_sigmaIetaIeta()

    def chargedHadronIso(self):
        return self.physObj.chargedHadronIso()

    def neutralHadronIso(self):
        return self.physObj.neutralHadronIso()

    def photonIso(self):
        return self.physObj.photonIso()

    def photonIDCSA14(self, name):
        keepThisPhoton = True
        if name == "PhotonCutBasedIDLoose_CSA14":
            if abs(self.physObj.eta())<1.479 :
                if self.full5x5_sigmaIetaIeta() > 0.012 : keepThisPhoton = False
                if self.hOVERe() > 0.0559       : keepThisPhoton = False
            else :
                if self.full5x5_sigmaIetaIeta() > 0.035 : keepThisPhoton = False
                if self.hOVERe() > 0.049        : keepThisPhoton = False
        return keepThisPhoton
                
    pass
