from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *

import ROOT

class Photon(PhysicsObject ):

    '''                                                                                                                                                                                                                                                                return object from the photon 
    '''
    def hOVERe(self):
        return self.physObj.hadTowOverEm() 

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
                if self.full5x5_sigmaIetaIeta() > 0.015 : keepThisPhoton = False
                if self.hOVERe() > 0.0559       : keepThisPhoton = False
            else :
                if self.full5x5_sigmaIetaIeta() > 0.035 : keepThisPhoton = False
                if self.hOVERe() > 0.049        : keepThisPhoton = False
        return keepThisPhoton

    def CutBasedIDWP(self,name):
        # recommeneded PHYS14 working points from POG
        WPs = {
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2#Pointers_for_PHYS14_selection_im
        "POG_PHYS14_25ns_Loose": {"conversionVeto": [True,True], "H/E":[0.048,0.069],"sigmaIEtaIEta":[0.0106,0.0266],
        "chaHadIso":[2.56,3.12],"neuHadIso":[[3.74,0.0025],[17.11,0.0118]],"phoIso":[[2.68,0.001],[2.70,0.0059]]},
        
        "POG_PHYS14_25ns_Medium": {"conversionVeto": [True,True], "H/E":[0.032,0.0166],"sigmaIEtaIEta":[0.0101,0.0264],
        "chaHadIso":[1.90,1.95],"neuHadIso":[[2.96,0.0025],[4.42,0.0118]],"phoIso":[[1.39,0.001],[1.89,0.0059]]},
        
        "POG_PHYS14_25ns_Tight": {"conversionVeto": [True,True], "H/E":[0.011,0.015],"sigmaIEtaIEta":[0.0099,0.0263],
        "chaHadIso":[1.86,1.68],"neuHadIso":[[2.64,0.0025],[4.42,0.0118]],"phoIso":[[1.39,0.001],[1.03,0.0059]]},

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2#CSA14_selections_for_20_bx_25_sc
        "POG_CSA14_25ns_Loose": {"conversionVeto": [True,True], "H/E":[0.553,0.062],"sigmaIEtaIEta":[0.0099,0.0284],
        "chaHadIso":[2.49,1.04],"neuHadIso":[[15.43,0.007],[19.71,0.0129]],"phoIso":[[9.42,0.0033],[11.88,0.0108]]},
        
        "POG_CSA14_25ns_Medium": {"conversionVeto": [True,True], "H/E":[0.058,0.020],"sigmaIEtaIEta":[0.0099,0.0268],
        "chaHadIso":[1.91,0.82],"neuHadIso":[[4.66,0.007],[14.65,0.0129]],"phoIso":[[4.29,0.0033],[4.06,0.0108]]},
        
        "POG_CSA14_25ns_Tight": {"conversionVeto": [True,True], "H/E":[0.019,0.016],"sigmaIEtaIEta":[0.0099,0.0263],
        "chaHadIso":[1.61,0.69],"neuHadIso":[[3.98,0.007],[4.52,0.0129]],"phoIso":[[3.01,0.0033],[3.61,0.0108]]},
        }
        return WPs[name]


    def etaRegionID(self):
        #return 0 if the photon is in barrel and 1 if in endcap
        if abs(self.physObj.eta())<1.479 :
            idForBarrel = 0
        else:
            idForBarrel = 1
        return idForBarrel

    def calScaledIsoValue(self,slope,offset):
        return slope*self.pt()+offset


    def passPhotonID(self,name):

        idForBarrel = self.etaRegionID()
        passPhotonID = True

        if self.CutBasedIDWP(name)["conversionVeto"][idForBarrel] and self.physObj.hasPixelSeed():
            passPhotonID = False

        if self.CutBasedIDWP(name)["H/E"][idForBarrel] < self.hOVERe():
            passPhotonID = False

        if self.CutBasedIDWP(name)["sigmaIEtaIEta"][idForBarrel] < self.full5x5_sigmaIetaIeta():
            passPhotonID = False

        if self.CutBasedIDWP(name)["chaHadIso"][idForBarrel] < self.chargedHadronIso():
            passPhotonID = False

        if self.calScaledIsoValue(*self.CutBasedIDWP(name)["neuHadIso"][idForBarrel]) < self.neutralHadronIso():
            passPhotonID = False

        if self.calScaledIsoValue(*self.CutBasedIDWP(name)["phoIso"][idForBarrel]) < self.photonIso():
            passPhotonID = False
        
        return passPhotonID


                
    pass

setattr(ROOT.pat.Photon, "recoPhotonIso", ROOT.reco.Photon.photonIso)
setattr(ROOT.pat.Photon, "recoNeutralHadronIso", ROOT.reco.Photon.neutralHadronIso)
setattr(ROOT.pat.Photon, "recoChargedHadronIso", ROOT.reco.Photon.chargedHadronIso)
