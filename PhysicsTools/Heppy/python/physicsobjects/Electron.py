from PhysicsTools.Heppy.physicsobjects.Lepton import Lepton
from PhysicsTools.Heppy.physicsutils.ElectronMVAID import *
import ROOT

class Electron( Lepton ):

    def __init__(self, *args, **kwargs):
        '''Initializing tightIdResult to None. The user is responsible
        for setting this attribute externally if he wants to use the tightId
        function.'''
        super(Electron, self).__init__(*args, **kwargs)
        self._physObjInit()

    def _physObjInit(self):
        self.tightIdResult = None
        self.associatedVertex = None
        self.rho              = None
        self._mvaNonTrigV0  = {True:None, False:None}
        self._mvaTrigV0     = {True:None, False:None}
        self._mvaTrigNoIPV0 = {True:None, False:None}
        self._mvaRun2 = {}

    def electronID( self, id, vertex=None, rho=None ):
        if id is None or id == "": return True
        if vertex == None and hasattr(self,'associatedVertex') and self.associatedVertex != None: vertex = self.associatedVertex
        if rho == None and hasattr(self,'rho') and self.rho != None: rho = self.rho
        if   id == "POG_MVA_ID_NonTrig":  return self.mvaIDLoose()
        elif id == "POG_MVA_ID_Trig":     return self.mvaIDTight()
        elif id == "POG_MVA_ID_NonTrig_full5x5":  return self.mvaIDLoose(full5x5=True)
        elif id == "POG_MVA_ID_Trig_full5x5":     return self.mvaIDTight(full5x5=True)
        elif id == "POG_MVA_ID_Run2_NonTrig_VLoose":   return self.mvaIDRun2("NonTrigPhys14","VLoose")
        elif id == "POG_MVA_ID_Run2_NonTrig_Loose":    return self.mvaIDRun2("NonTrigPhys14","Loose")
        elif id == "POG_MVA_ID_Run2_NonTrig_Tight":    return self.mvaIDRun2("NonTrigPhys14","Tight")
        elif id.startswith("POG_Cuts_ID_"):
                return self.cutBasedId(id.replace("POG_Cuts_ID_","POG_"))
        for ID in self.electronIDs():
            if ID.first == id:
                return ID.second
        raise RuntimeError, "Electron id '%s' not yet implemented in Electron.py" % id

    def cutBasedId(self, wp, showerShapes="auto"):
        if "_full5x5" in wp:
            showerShapes = "full5x5"
            wp = wp.replace("_full5x5","")
        elif showerShapes == "auto":
            if "POG_CSA14_25ns_v1" in wp or "POG_CSA14_50ns_v1" in wp or "POG_PHYS14_25ns_v1" in wp or "POG_PHYS14_25ns_v1_ConvVeto" in wp or "POG_PHYS14_25ns_v1_ConvVetoDxyDz" in wp:
                showerShapes = "full5x5"
        vars = {
            'dEtaIn' : abs(self.physObj.deltaEtaSuperClusterTrackAtVtx()),
            'dPhiIn' : abs(self.physObj.deltaPhiSuperClusterTrackAtVtx()),
            'sigmaIEtaIEta' : self.physObj.full5x5_sigmaIetaIeta() if showerShapes == "full5x5" else self.physObj.sigmaIetaIeta(),
            'H/E' : self.physObj.hadronicOverEm(),
            #'1/E-1/p' : abs(1.0/self.physObj.ecalEnergy() - self.physObj.eSuperClusterOverP()/self.physObj.ecalEnergy()),
            '1/E-1/p' : abs(1.0/self.physObj.ecalEnergy() - self.physObj.eSuperClusterOverP()/self.physObj.ecalEnergy()) if self.physObj.ecalEnergy()>0. else 9e9,
            'conversionVeto' : self.physObj.passConversionVeto(),
            'missingHits' : self.physObj.gsfTrack().hitPattern().numberOfHits(ROOT.reco.HitPattern.MISSING_INNER_HITS), # http://cmslxr.fnal.gov/source/DataFormats/TrackReco/interface/HitPattern.h?v=CMSSW_7_2_3#0153
            'dxy' : abs(self.dxy()),
            'dz' : abs(self.dz()),
        }
        WP = {
            ## ------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaCutBasedIdentification?rev=31
            'POG_2012_Veto'   :  [('dEtaIn', [0.007, 0.01]),  ('dPhiIn', [0.8,  0.7 ]), ('sigmaIEtaIEta', [0.01, 0.03]), ('H/E', [0.15, 9e9]), ('1/E-1/p', [9e9,   9e9])],
            'POG_2012_Loose'  :  [('dEtaIn', [0.007, 0.009]), ('dPhiIn', [0.15, 0.1 ]), ('sigmaIEtaIEta', [0.01, 0.03]), ('H/E', [0.12, 0.1]), ('1/E-1/p', [0.05, 0.05])],
            'POG_2012_Medium' :  [('dEtaIn', [0.004, 0.007]), ('dPhiIn', [0.06, 0.03]), ('sigmaIEtaIEta', [0.01, 0.03]), ('H/E', [0.12, 0.1]), ('1/E-1/p', [0.05, 0.05])],
            'POG_2012_Tight'  :  [('dEtaIn', [0.004, 0.005]), ('dPhiIn', [0.03, 0.02]), ('sigmaIEtaIEta', [0.01, 0.03]), ('H/E', [0.12, 0.1]), ('1/E-1/p', [0.05, 0.05])],
            # RIC: in the EG POG WPs, isolation is included too. Here only the pure ID part.
            # dz and d0 cuts are excluded here as well.
            ## ------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2#Working_points_for_CSA14_samples?rev=13
            'POG_CSA14_25ns_v1_Veto'   :  [('dEtaIn', [0.017938, 0.014569]), ('dPhiIn', [0.182958, 0.230914]), ('sigmaIEtaIEta', [0.012708, 0.036384]), ('H/E', [0.335015, 0.200792]), ('1/E-1/p', [0.198287, 0.146856])],
            'POG_CSA14_25ns_v1_Loose'  :  [('dEtaIn', [0.014928, 0.013045]), ('dPhiIn', [0.141050, 0.149017]), ('sigmaIEtaIEta', [0.011304, 0.035536]), ('H/E', [0.127690, 0.107898]), ('1/E-1/p', [0.097806, 0.102261])],
            'POG_CSA14_25ns_v1_Medium' :  [('dEtaIn', [0.013071, 0.010006]), ('dPhiIn', [0.132113, 0.052321]), ('sigmaIEtaIEta', [0.010726, 0.032882]), ('H/E', [0.109761, 0.101755]), ('1/E-1/p', [0.032639, 0.041427])],
            'POG_CSA14_25ns_v1_Tight'  :  [('dEtaIn', [0.012671, 0.008823]), ('dPhiIn', [0.025218, 0.027286]), ('sigmaIEtaIEta', [0.010061, 0.030222]), ('H/E', [0.065085, 0.090710]), ('1/E-1/p', [0.027873, 0.019404])],
            'POG_CSA14_50ns_v1_Veto'   :  [('dEtaIn', [0.021, 0.028]), ('dPhiIn', [0.25 , 0.23 ]), ('sigmaIEtaIEta', [0.012, 0.035]), ('H/E', [0.24 , 0.19 ]), ('1/E-1/p', [0.32 , 0.13 ])],
            'POG_CSA14_50ns_v1_Loose'  :  [('dEtaIn', [0.016, 0.025]), ('dPhiIn', [0.080, 0.097]), ('sigmaIEtaIEta', [0.012, 0.032]), ('H/E', [0.15 , 0.12 ]), ('1/E-1/p', [0.11 , 0.11 ])],
            'POG_CSA14_50ns_v1_Medium' :  [('dEtaIn', [0.015, 0.023]), ('dPhiIn', [0.051, 0.056]), ('sigmaIEtaIEta', [0.010, 0.030]), ('H/E', [0.10 , 0.099]), ('1/E-1/p', [0.053, 0.11 ])],
            'POG_CSA14_50ns_v1_Tight'  :  [('dEtaIn', [0.012, 0.019]), ('dPhiIn', [0.024, 0.043]), ('sigmaIEtaIEta', [0.010, 0.029]), ('H/E', [0.074, 0.080]), ('1/E-1/p', [0.026, 0.076])],
            ## ------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2#Working_points_for_PHYS14_sample?rev=13
            'POG_PHYS14_25ns_v1_Veto'   :  [('dEtaIn', [0.016315, 0.010671]), ('dPhiIn', [0.252044, 0.245263]), ('sigmaIEtaIEta', [0.011100 , 0.033987]), ('H/E', [0.345843, 0.134691]), ('1/E-1/p', [0.248070, 0.157160])],
            'POG_PHYS14_25ns_v1_Loose'  :  [('dEtaIn', [0.012442, 0.010654]), ('dPhiIn', [0.072624, 0.145129]), ('sigmaIEtaIEta', [0.010557 , 0.032602]), ('H/E', [0.121476, 0.131862]), ('1/E-1/p', [0.221803, 0.142283])],
            'POG_PHYS14_25ns_v1_Medium' :  [('dEtaIn', [0.007641, 0.009285]), ('dPhiIn', [0.032643, 0.042447]), ('sigmaIEtaIEta', [0.010399 , 0.029524]), ('H/E', [0.060662, 0.104263]), ('1/E-1/p', [0.153897, 0.137468])],
            'POG_PHYS14_25ns_v1_Tight'  :  [('dEtaIn', [0.006574, 0.005681]), ('dPhiIn', [0.022868, 0.032046]), ('sigmaIEtaIEta', [0.010181 , 0.028766]), ('H/E', [0.037553, 0.081902]), ('1/E-1/p', [0.131191, 0.106055])],
        }
        WP_conversion_veto = {
            # missing Hits incremented by 1 because we return False if >=, note the '='
            ## ------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2#Working_points_for_CSA14_samples?rev=13
            'POG_CSA14_25ns_v1_ConvVeto_Veto'   :  WP['POG_CSA14_25ns_v1_Veto'  ]+[('conversionVeto', [True, True]), ('missingHits', [3, 4])],
            'POG_CSA14_25ns_v1_ConvVeto_Loose'  :  WP['POG_CSA14_25ns_v1_Loose' ]+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
            'POG_CSA14_25ns_v1_ConvVeto_Medium' :  WP['POG_CSA14_25ns_v1_Medium']+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
            'POG_CSA14_25ns_v1_ConvVeto_Tight'  :  WP['POG_CSA14_25ns_v1_Tight' ]+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
            'POG_CSA14_50ns_v1_ConvVeto_Veto'   :  WP['POG_CSA14_50ns_v1_Veto'  ]+[('conversionVeto', [True, True]), ('missingHits', [3, 4])],
            'POG_CSA14_50ns_v1_ConvVeto_Loose'  :  WP['POG_CSA14_50ns_v1_Loose' ]+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
            'POG_CSA14_50ns_v1_ConvVeto_Medium' :  WP['POG_CSA14_50ns_v1_Medium']+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
            'POG_CSA14_50ns_v1_ConvVeto_Tight'  :  WP['POG_CSA14_50ns_v1_Tight' ]+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
            ## ------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2#Working_points_for_PHYS14_sample?rev=13
            'POG_PHYS14_25ns_v1_ConvVeto_Veto'   :  WP['POG_PHYS14_25ns_v1_Veto'  ]+[('conversionVeto', [True, True]), ('missingHits', [3, 4])],
            'POG_PHYS14_25ns_v1_ConvVeto_Loose'  :  WP['POG_PHYS14_25ns_v1_Loose' ]+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
            'POG_PHYS14_25ns_v1_ConvVeto_Medium' :  WP['POG_PHYS14_25ns_v1_Medium']+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
            'POG_PHYS14_25ns_v1_ConvVeto_Tight'  :  WP['POG_PHYS14_25ns_v1_Tight' ]+[('conversionVeto', [True, True]), ('missingHits', [2, 2])],
        }

        WP.update(WP_conversion_veto)

        WP_conversion_veto_DxyDz = {
            # missing Hits incremented by 1 because we return False if >=, note the '='
            ## ------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2#Working_points_for_PHYS14_sample
            'POG_PHYS14_25ns_v1_ConvVetoDxyDz_Veto'   :  WP['POG_PHYS14_25ns_v1_ConvVeto_Veto'  ]+[('dxy',[0.060279, 0.273097]), ('dz',[0.800538, 0.885860])],
            'POG_PHYS14_25ns_v1_ConvVetoDxyDz_Loose'  :  WP['POG_PHYS14_25ns_v1_ConvVeto_Loose' ]+[('dxy',[0.022664, 0.097358]), ('dz',[0.173670, 0.198444])],
            'POG_PHYS14_25ns_v1_ConvVetoDxyDz_Medium' :  WP['POG_PHYS14_25ns_v1_ConvVeto_Medium']+[('dxy',[0.011811, 0.051682]), ('dz',[0.070775, 0.180720])],
            'POG_PHYS14_25ns_v1_ConvVetoDxyDz_Tight'  :  WP['POG_PHYS14_25ns_v1_ConvVeto_Tight' ]+[('dxy',[0.009924, 0.027261]), ('dz',[0.015310, 0.147154])],
        }

        WP.update(WP_conversion_veto_DxyDz)


        if wp not in WP:
            raise RuntimeError, "Working point '%s' not yet implemented in Electron.py" % wp
        for (cut_name,(cut_eb,cut_ee)) in WP[wp]:
            if cut_name == 'conversionVeto':
                if (cut_eb if self.physObj.isEB() else cut_ee) and not vars[cut_name]:
                    return False
            elif vars[cut_name] >= (cut_eb if self.physObj.isEB() else cut_ee):
                return False
        return True

    def absEffAreaIso(self,rho,effectiveAreas):
        '''MIKE, missing doc.
        Should have the same name as the function in the mother class.
        Can call the mother class function with super.
        '''
        return self.absIsoFromEA(rho,self.superCluster().eta(),effectiveAreas.eGamma)

    def mvaId( self ):
        return self.mvaNonTrigV0()

    def tightId( self ):
        return self.tightIdResult

    def mvaNonTrigV0( self, full5x5=False, debug = False ):
        if self._mvaNonTrigV0[full5x5] == None:
            if self.associatedVertex == None: raise RuntimeError, "You need to set electron.associatedVertex before calling any MVA"
            if self.rho              == None: raise RuntimeError, "You need to set electron.rho before calling any MVA"
            self._mvaNonTrigV0[full5x5] = ElectronMVAID_NonTrig(self.physObj, self.associatedVertex, self.rho, full5x5, debug)
        return self._mvaNonTrigV0[full5x5]

    def mvaTrigV0( self, full5x5=False, debug = False ):
        if self._mvaTrigV0[full5x5] == None:
            if self.associatedVertex == None: raise RuntimeError, "You need to set electron.associatedVertex before calling any MVA"
            if self.rho              == None: raise RuntimeError, "You need to set electron.rho before calling any MVA"
            self._mvaTrigV0[full5x5] = ElectronMVAID_Trig(self.physObj, self.associatedVertex, self.rho, full5x5, debug)
        return self._mvaTrigV0[full5x5]

    def mvaTrigNoIPV0( self, full5x5=False, debug = False ):
        if self._mvaTrigNoIPV0[full5x5] == None:
            if self.associatedVertex == None: raise RuntimeError, "You need to set electron.associatedVertex before calling any MVA"
            if self.rho              == None: raise RuntimeError, "You need to set electron.rho before calling any MVA"
            self._mvaTrigNoIPV0[full5x5] = ElectronMVAID_TrigNoIP(self.physObj, self.associatedVertex, self.rho, full5x5, debug)
        return self._mvaTrigNoIPV0[full5x5]

    def mvaRun2( self, name, debug = False ):
        if name not in self._mvaRun2:
            if name not in ElectronMVAID_ByName: raise RuntimeError, "Unknown electron run2 mva id %s (known ones are: %s)\n" % (name, ElectronMVAID_ByName.keys())
            if self.associatedVertex == None: raise RuntimeError, "You need to set electron.associatedVertex before calling any MVA"
            if self.rho              == None: raise RuntimeError, "You need to set electron.rho before calling any MVA"
            self._mvaRun2[name] = ElectronMVAID_ByName[name](self.physObj, self.associatedVertex, self.rho, True, debug)
        return self._mvaRun2[name]

    def mvaIDTight(self, full5x5=False):
            eta = abs(self.superCluster().eta())
            if self.pt() < 20:
                if   (eta < 0.8)  : return self.mvaTrigV0(full5x5) > +0.00;
                elif (eta < 1.479): return self.mvaTrigV0(full5x5) > +0.10;
                else              : return self.mvaTrigV0(full5x5) > +0.62;
            else:
                if   (eta < 0.8)  : return self.mvaTrigV0(full5x5) > +0.94;
                elif (eta < 1.479): return self.mvaTrigV0(full5x5) > +0.85;
                else              : return self.mvaTrigV0(full5x5) > +0.92;

    def mvaIDLoose(self, full5x5=False):
            eta = abs(self.superCluster().eta())
            if self.pt() < 10:
                if   (eta < 0.8)  : return self.mvaNonTrigV0(full5x5) > +0.47;
                elif (eta < 1.479): return self.mvaNonTrigV0(full5x5) > +0.004;
                else              : return self.mvaNonTrigV0(full5x5) > +0.295;
            else:
                if   (eta < 0.8)  : return self.mvaNonTrigV0(full5x5) > -0.34;
                elif (eta < 1.479): return self.mvaNonTrigV0(full5x5) > -0.65;
                else              : return self.mvaNonTrigV0(full5x5) > +0.60;

    def mvaIDRun2(self, name, wp):
            eta = abs(self.superCluster().eta())
            if name == "NonTrigPhys14":
                if wp=="Loose":
                    if   (eta < 0.8)  : return self.mvaRun2(name) > +0.35;
                    elif (eta < 1.479): return self.mvaRun2(name) > +0.20;
                    else              : return self.mvaRun2(name) > -0.52;
                elif wp=="VLoose":
                    if   (eta < 0.8)  : return self.mvaRun2(name) > -0.11;
                    elif (eta < 1.479): return self.mvaRun2(name) > -0.35;
                    else              : return self.mvaRun2(name) > -0.55;
                elif wp=="Tight":
                    if   (eta < 0.8)  : return self.mvaRun2(name) > 0.73;
                    elif (eta < 1.479): return self.mvaRun2(name) > 0.57;
                    else              : return self.mvaRun2(name) > 0.05;
                else: raise RuntimeError, "Ele MVA ID Working point not found"
            else: raise RuntimeError, "Ele MVA ID type not found"


    def mvaIDZZ(self):
        return self.mvaIDLoose() and (self.gsfTrack().trackerExpectedHitsInner().numberOfLostHits()<=1)

    def chargedHadronIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumChargedHadronPt
        elif R == 0.4: return self.physObj.chargedHadronIso()
        raise RuntimeError, "Electron chargedHadronIso missing for R=%s" % R

    def neutralHadronIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumNeutralHadronEt
        elif R == 0.4: return self.physObj.neutralHadronIso()
        raise RuntimeError, "Electron neutralHadronIso missing for R=%s" % R

    def photonIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumPhotonEt
        elif R == 0.4: return self.physObj.photonIso()
        raise RuntimeError, "Electron photonIso missing for R=%s" % R

    def chargedAllIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumChargedParticlePt
        raise RuntimeError, "Electron chargedAllIso missing for R=%s" % R

    def chargedAllIso(self):
        raise RuntimeError, "Electron chargedAllIso missing"

    def puChargedHadronIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumPUPt
        elif R == 0.4: return self.physObj.puChargedHadronIso()
        raise RuntimeError, "Electron chargedHadronIso missing for R=%s" % R

    def dxy(self, vertex=None):
        '''Returns dxy.
        Computed using vertex (or self.associatedVertex if vertex not specified),
        and the gsf track.
        '''
        if vertex is None:
            vertex = self.associatedVertex
        return self.gsfTrack().dxy( vertex.position() )

    def edxy(self):
        '''returns the uncertainty on dxy (from gsf track)'''
        return self.gsfTrack().dxyError()

    def p4(self):
	 return ROOT.reco.Candidate.p4(self.physObj)

#    def p4(self):
#        return self.physObj.p4(self.physObj.candidateP4Kind()) # if kind == None else kind)

    def dz(self, vertex=None):
        '''Returns dz.
        Computed using vertex (or self.associatedVertex if vertex not specified),
        and the gsf track.
        '''
        if vertex is None:
            vertex = self.associatedVertex
        return self.gsfTrack().dz( vertex.position() )

    def edz(self):
        '''returns the uncertainty on dxz (from gsf track)'''
        return self.gsfTrack().dzError()


    def lostInner(self) :
        if hasattr(self.gsfTrack(),"trackerExpectedHitsInner") :
		return self.gsfTrack().trackerExpectedHitsInner().numberOfLostHits()
	else :
		return self.gsfTrack().hitPattern().numberOfHits(ROOT.reco.HitPattern.MISSING_INNER_HITS)

