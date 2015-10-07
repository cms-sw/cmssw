from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *
from math import exp
import re

import ROOT

class Photon(PhysicsObject ):

    def __init__(self, *args, **kwargs):
        '''Initializing rho to None. The user is responsible for setting it to the right value 
        to get the rho-corrected isolation.'''
        super(Photon, self).__init__(*args, **kwargs)
        self._physObjInit()

    def _physObjInit(self):
        self.rho = None


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

    def chargedHadronIso(self, corr=None):
        isoCharged = self.ftprAbsIsoCharged03 if hasattr(self,'ftprAbsIsoCharged03') else self.physObj.chargedHadronIso()
        if corr is None or corr == "": return isoCharged
        elif corr == "rhoArea": return max(isoCharged-self.rho*self.EffectiveArea03[0],0)
        else: raise RuntimeError, "Photon isolation correction '%s' not yet implemented in Photon.py" % corr

    def neutralHadronIso(self, corr=None):
        isoNHad = self.ftprAbsIsoNHad03 if hasattr(self,'ftprAbsIsoNHad03') else self.physObj.neutralHadronIso()
        if corr is None or corr == "": return isoNHad
        elif corr == "rhoArea": return max(isoNHad-self.rho*self.EffectiveArea03[1],0)
        else: raise RuntimeError, "Photon isolation correction '%s' not yet implemented in Photon.py" % corr

    def photonIso(self, corr=None):
        isoPho = self.ftprAbsIsoPho03 if hasattr(self,'ftprAbsIsoPho03') else self.physObj.photonIso()
        if corr is None or corr == "": return isoPho
        elif corr == "rhoArea": return max(isoPho-self.rho*self.EffectiveArea03[2],0)
        else: raise RuntimeError, "Photon isolation correction '%s' not yet implemented in Photon.py" % corr

    def photonIDCSA14(self, name, sidebands=False):
        keepThisPhoton = True
        sigmaThresh  = 999
        hovereThresh = 999
        if name == "PhotonCutBasedIDLoose_CSA14":
            if abs(self.physObj.eta())<1.479 :
                sigmaThresh  = 0.010
                hovereThresh = 0.0559
            else :
                sigmaThresh  = 0.030
                hovereThresh = 0.049
        elif name == "PhotonCutBasedIDLoose_PHYS14":
            if abs(self.physObj.eta())<1.479 :
                sigmaThresh  = 0.0106
                hovereThresh = 0.048
            else :
                sigmaThresh  = 0.0266
                hovereThresh = 0.069
        else :
          print "WARNING! Unkown photon ID! Will return true!" 
          return True

        if sidebands:
          if abs(self.physObj.eta())<1.479 :
            sigmaThresh = 0.015
          else :
            sigmaThresh = 0.035

        if self.full5x5_sigmaIetaIeta() > sigmaThresh  : keepThisPhoton = False
        if self.hOVERe()                > hovereThresh : keepThisPhoton = False

        return keepThisPhoton

    def CutBasedIDWP( self, name):
        # recommeneded PHYS14 working points from POG
        WPs = {
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2#Pointers_for_PHYS14_selection_im
        "POG_PHYS14_25ns_Loose": {"conversionVeto": [True,True], "H/E":[0.028,0.093],"sigmaIEtaIEta":[0.0107,0.0272],
        "chaHadIso":[2.67,1.79],"neuHadIso":[[7.23,0.0028,0.5408],[8.89,0.01725]],"phoIso":[[2.11,0.0014],[3.09,0.0091]]},
        
        # https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedPhotonIdentificationRun2?rev=11
        "POG_PHYS14_25ns_Loose_old": {"conversionVeto": [True,True], "H/E":[0.048,0.069],"sigmaIEtaIEta":[0.0106,0.0266],
        "chaHadIso":[2.56,3.12],"neuHadIso":[[3.74,0.0025,0.],[17.11,0.0118,0.]],"phoIso":[[2.68,0.001],[2.70,0.0059]]},
        
        "POG_PHYS14_25ns_Medium": {"conversionVeto": [True,True], "H/E":[0.012,0.023],"sigmaIEtaIEta":[0.0100,0.0267],
        "chaHadIso":[1.79,1.09],"neuHadIso":[[0.16,0.0028,0.5408],[4.31,0.0172]],"phoIso":[[1.90,0.0014],[1.90,0.0091]]},
        
        "POG_PHYS14_25ns_Tight": {"conversionVeto": [True,True], "H/E":[0.010,0.015],"sigmaIEtaIEta":[0.0100,0.0265],
        "chaHadIso":[1.66,1.04],"neuHadIso":[[0.14,0.0028,0.5408],[3.89,0.0172]],"phoIso":[[1.40,0.0014],[1.40,0.0091]]},

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedPhotonIdentificationRun2#CSA14_selections_for_20_bx_25_sc
        "POG_CSA14_25ns_Loose": {"conversionVeto": [True,True], "H/E":[0.553,0.062],"sigmaIEtaIEta":[0.0099,0.0284],
        "chaHadIso":[2.49,1.04],"neuHadIso":[[15.43,0.007],[19.71,0.0129]],"phoIso":[[9.42,0.0033],[11.88,0.0108]]},
        
        "POG_CSA14_25ns_Medium": {"conversionVeto": [True,True], "H/E":[0.058,0.020],"sigmaIEtaIEta":[0.0099,0.0268],
        "chaHadIso":[1.91,0.82],"neuHadIso":[[4.66,0.007],[14.65,0.0129]],"phoIso":[[4.29,0.0033],[4.06,0.0108]]},
        
        "POG_CSA14_25ns_Tight": {"conversionVeto": [True,True], "H/E":[0.019,0.016],"sigmaIEtaIEta":[0.0099,0.0263],
        "chaHadIso":[1.61,0.69],"neuHadIso":[[3.98,0.007],[4.52,0.0129]],"phoIso":[[3.01,0.0033],[3.61,0.0108]]},
        }
        
        baseWP = re.split('_',name)
        if "looseSieie" in baseWP[-1]: 
            baseWP.pop()
            WPs["_".join(baseWP)]["sigmaIEtaIEta"] = [0.015,0.035]

        return WPs["_".join(baseWP)]


    def etaRegionID(self):
        #return 0 if the photon is in barrel and 1 if in endcap
        if abs(self.physObj.eta())<1.479 :
            idForBarrel = 0
        else:
            idForBarrel = 1
        return idForBarrel

    def calScaledIsoValueLin(self,offset,slope):
        return slope*self.pt()+offset

    def calScaledIsoValueExp(self,offset,slope_exp,offset_exp):
        return offset + exp(slope_exp*self.pt()+offset_exp)


    def passPhotonID(self,name):
        
        idForBarrel = self.etaRegionID()
        passPhotonID = True

        if self.CutBasedIDWP(name)["conversionVeto"][idForBarrel] and self.physObj.hasPixelSeed():
            passPhotonID = False

        if self.CutBasedIDWP(name)["H/E"][idForBarrel] < self.hOVERe():
            passPhotonID = False

        if self.CutBasedIDWP(name)["sigmaIEtaIEta"][idForBarrel] < self.full5x5_sigmaIetaIeta():
            passPhotonID = False

        return passPhotonID

    def passPhotonIso(self,name,isocorr):

        idForBarrel = self.etaRegionID()
        passPhotonIso = True

        if self.CutBasedIDWP(name)["chaHadIso"][idForBarrel] < self.chargedHadronIso(isocorr):
            passPhotonIso = False

        if "POG_PHYS14_25ns" in name and idForBarrel == 0:
            if self.calScaledIsoValueExp(*self.CutBasedIDWP(name)["neuHadIso"][idForBarrel]) < self.neutralHadronIso(isocorr):
                passPhotonIso = False
        else:
            if self.calScaledIsoValueLin(*self.CutBasedIDWP(name)["neuHadIso"][idForBarrel]) < self.neutralHadronIso(isocorr):
                passPhotonIso = False

        if self.calScaledIsoValueLin(*self.CutBasedIDWP(name)["phoIso"][idForBarrel]) < self.photonIso(isocorr):
            passPhotonIso = False
        
        return passPhotonIso

    pass

setattr(ROOT.pat.Photon, "recoPhotonIso", ROOT.reco.Photon.photonIso)
setattr(ROOT.pat.Photon, "recoNeutralHadronIso", ROOT.reco.Photon.neutralHadronIso)
setattr(ROOT.pat.Photon, "recoChargedHadronIso", ROOT.reco.Photon.chargedHadronIso)
