from PhysicsTools.Heppy.physicsobjects.Lepton import Lepton
from PhysicsTools.Heppy.physicsobjects.ElectronMVAID import ElectronMVAID_Trig, ElectronMVAID_NonTrig, ElectronMVAID_TrigNoIP


class Electron( Lepton ):

    def __init__(self, *args, **kwargs):
        '''Initializing tightIdResult to None. The user is responsible
        for setting this attribute externally if he wants to use the tightId
        function.'''
        super(Electron, self).__init__(*args, **kwargs)
        self.tightIdResult = None
        self.associatedVertex = None
        self.rho              = None
        self._mvaNonTrigV0  = {True:None, False:None}
        self._mvaTrigV0     = {True:None, False:None}
        self._mvaTrigNoIPV0 = {True:None, False:None}

    def electronID( self, id, vertex=None, rho=None ):
        if id is None or id == "": return True
        if vertex == None and hasattr(self,'associatedVertex') and self.associatedVertex != None: vertex = self.associatedVertex
        if rho == None and hasattr(self,'rho') and self.rho != None: rho = self.rho
        if   id == "POG_MVA_ID_NonTrig":  return self.mvaIDLoose()
        elif id == "POG_MVA_ID_Trig":     return self.mvaIDTight()
        elif id == "POG_MVA_ID_NonTrig_full5x5":  return self.mvaIDLoose(full5x5=True)
        elif id == "POG_MVA_ID_Trig_full5x5":     return self.mvaIDTight(full5x5=True)
        elif id.startswith("POG_Cuts_ID_"): 
                return self.cutBasedId(id.replace("POG_Cuts_ID_","POG_")) 
        raise RuntimeError, "Electron id '%s' not yet implemented in Electron.py" % id

    def cutBasedId(self, wp, showerShapes="auto"):
        if "_full5x5" in wp:
            showerShapes = "full5x5"
            wp = wp.replace("_full5x5","")
        elif showerShapes == "auto":
            if "POG_CSA14_25ns_v1" in wp or "POG_CSA14_50ns_v1" in wp:
                showerShapes = "full5x5"
        vars = {
            'dEtaIn' : abs(self.physObj.deltaEtaSuperClusterTrackAtVtx()),
            'dPhiIn' : abs(self.physObj.deltaPhiSuperClusterTrackAtVtx()),
            'sigmaIEtaIEta' : self.physObj.full5x5_sigmaIetaIeta() if showerShapes == "full5x5" else self.physObj.sigmaIetaIeta(),
            'H/E' : self.physObj.hadronicOverEm(),
            #'1/E-1/p' : abs(1.0/self.physObj.ecalEnergy() - self.physObj.eSuperClusterOverP()/self.physObj.ecalEnergy()),
            '1/E-1/p' : abs(1.0/self.physObj.ecalEnergy() - self.physObj.eSuperClusterOverP()/self.physObj.ecalEnergy()) if self.physObj.ecalEnergy()>0. else 9e9,
        }
        WP = {
            ## ------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaCutBasedIdentification?rev=31
            'POG_2012_Veto'   :  [('dEtaIn', [0.007, 0.01]),  ('dPhiIn', [0.8,  0.7 ]), ('sigmaIEtaIEta', [0.01, 0.03]), ('H/E', [0.15, 9e9]), ('1/E-1/p', [9e9,   9e9])],
            'POG_2012_Loose'  :  [('dEtaIn', [0.007, 0.009]), ('dPhiIn', [0.15, 0.1 ]), ('sigmaIEtaIEta', [0.01, 0.03]), ('H/E', [0.12, 0.1]), ('1/E-1/p', [0.05, 0.05])],
            'POG_2012_Medium' :  [('dEtaIn', [0.004, 0.007]), ('dPhiIn', [0.06, 0.03]), ('sigmaIEtaIEta', [0.01, 0.03]), ('H/E', [0.12, 0.1]), ('1/E-1/p', [0.05, 0.05])],
            'POG_2012_Tight'  :  [('dEtaIn', [0.004, 0.005]), ('dPhiIn', [0.03, 0.02]), ('sigmaIEtaIEta', [0.01, 0.03]), ('H/E', [0.12, 0.1]), ('1/E-1/p', [0.05, 0.05])],
            ## ------- https://indico.cern.cH/Event/298242/contribution/1/material/slides/5.pdf (slide 5)
            'POG_CSA14_25ns_v1_Veto'   :  [('dEtaIn', [0.012, 0.015]), ('dPhiIn', [0.8,  0.7 ]), ('sigmaIEtaIEta', [0.01 , 0.033]), ('H/E', [0.15, 9e9 ]), ('1/E-1/p', [9e9,   9e9])],
            'POG_CSA14_25ns_v1_Loose'  :  [('dEtaIn', [0.012, 0.014]), ('dPhiIn', [0.15, 0.1 ]), ('sigmaIEtaIEta', [0.01 , 0.033]), ('H/E', [0.12, 0.12]), ('1/E-1/p', [0.05, 0.05])],
            'POG_CSA14_25ns_v1_Medium' :  [('dEtaIn', [0.009, 0.012]), ('dPhiIn', [0.06, 0.03]), ('sigmaIEtaIEta', [0.01 , 0.031]), ('H/E', [0.12, 0.12]), ('1/E-1/p', [0.05, 0.05])],
            'POG_CSA14_25ns_v1_Tight'  :  [('dEtaIn', [0.009, 0.010]), ('dPhiIn', [0.03, 0.02]), ('sigmaIEtaIEta', [0.01 , 0.031]), ('H/E', [0.12, 0.12]), ('1/E-1/p', [0.05, 0.05])],
            'POG_CSA14_50ns_v1_Veto'   :  [('dEtaIn', [0.012, 0.022]), ('dPhiIn', [0.8,  0.7 ]), ('sigmaIEtaIEta', [0.012, 0.033]), ('H/E', [0.15, 9e9 ]), ('1/E-1/p', [9e9,   9e9])],
            'POG_CSA14_50ns_v1_Loose'  :  [('dEtaIn', [0.012, 0.021]), ('dPhiIn', [0.15, 0.1 ]), ('sigmaIEtaIEta', [0.012, 0.033]), ('H/E', [0.12, 0.12]), ('1/E-1/p', [0.05, 0.05])],
            'POG_CSA14_50ns_v1_Medium' :  [('dEtaIn', [0.009, 0.019]), ('dPhiIn', [0.06, 0.03]), ('sigmaIEtaIEta', [0.01 , 0.031]), ('H/E', [0.12, 0.12]), ('1/E-1/p', [0.05, 0.05])],
            'POG_CSA14_50ns_v1_Tight'  :  [('dEtaIn', [0.009, 0.017]), ('dPhiIn', [0.03, 0.02]), ('sigmaIEtaIEta', [0.01 , 0.031]), ('H/E', [0.12, 0.12]), ('1/E-1/p', [0.05, 0.05])],
        }
        if wp not in WP: 
            raise RuntimeError, "Working point '%s' not yet implemented in Electron.py" % wp
        for (cut_name,(cut_eb,cut_ee)) in WP[wp]:
            if vars[cut_name] >= (cut_eb if self.physObj.isEB() else cut_ee):
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

    def mvaIDZZ(self):
        return self.mvaIDLoose() and (self.gsfTrack().trackerExpectedHitsInner().numberOfLostHits()<=1)

    def chargedHadronIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumChargedHadronPt 
        elif R == 0.4: return self.physObj.chargedHadronIso()
        raise RuntimeError, "Electron chargedHadronIso missing for R=%s" % R

    def neutralHadronIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumNeutralHadronEt 
        elif R == 0.4: return self.physObj.neutralHadronIso()
        raise RuntimeError, "Electron neutralHadronIso missing for R=%s" % R

    def photonIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumPhotonEt 
        elif R == 0.4: return self.physObj.photonIso()
        raise RuntimeError, "Electron photonIso missing for R=%s" % R

    def chargedAllIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumChargedParticlePt 
        raise RuntimeError, "Electron chargedAllIso missing for R=%s" % R

    def puChargedHadronIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationVariables().sumPUPt 
        elif R == 0.4: return self.physObj.puChargedHadronIso()
        raise RuntimeError, "Electron chargedHadronIso missing for R=%s" % R




    def chargedAllIso(self):
        '''This function is used in the isolation, see Lepton class.
        Here, we replace the all charged isolation by the all charged isolation with cone veto'''
        return self.chargedAllIsoWithConeVeto()


    def dxy(self, vertex=None):
        '''Returns dxy.
        Computed using vertex (or self.associatedVertex if vertex not specified),
        and the gsf track.
        '''
        if vertex is None:
            vertex = self.associatedVertex
        return self.gsfTrack().dxy( vertex.position() )
 

    def p4(self,kind=None):
        return self.physObj.p4(self.physObj.candidateP4Kind() if kind == None else kind)

    def dz(self, vertex=None):
        '''Returns dz.
        Computed using vertex (or self.associatedVertex if vertex not specified),
        and the gsf track.
        '''
        if vertex is None:
            vertex = self.associatedVertex
        return self.gsfTrack().dz( vertex.position() )


