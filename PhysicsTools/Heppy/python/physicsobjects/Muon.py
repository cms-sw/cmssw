from PhysicsTools.Heppy.physicsobjects.Lepton import Lepton
from PhysicsTools.HeppyCore.utils.deltar import deltaR

class Muon( Lepton ):
    def __init__(self, *args, **kwargs):
        super(Muon, self).__init__(*args, **kwargs)
        self._trackForDxyDz = "muonBestTrack"

    def setTrackForDxyDz(self,what):
        if not hasattr(self,what):
            raise RuntimeError, "I don't have a track called "+what
        self._trackForDxyDz = what

    def looseId( self ):
        '''Loose ID as recommended by mu POG.'''
        return self.physObj.isLooseMuon()

    def tightId( self ):
        '''Tight ID as recommended by mu POG (unless redefined in the lepton analyzer).'''
        return getattr(self,"tightIdResult",self.muonID("POG_ID_Tight"))

    def muonID(self, name, vertex=None):
        if name == "" or name is None: 
            return True
        if name.startswith("POG_"):
            if name == "POG_ID_Loose": return self.physObj.isLooseMuon()
            if vertex is None:
                vertex = getattr(self, 'associatedVertex', None)
            if name == "POG_ID_Tight":  return self.physObj.isTightMuon(vertex)
            if name == "POG_ID_HighPt": return self.physObj.isHighPtMuon(vertex)
            if name == "POG_ID_Soft":   return self.physObj.isSoftMuon(vertex)
            if name == "POG_ID_TightNoVtx":  return self.looseId() and \
                                                 self.isGlobalMuon() and \
                                                 self.globalTrack().normalizedChi2() < 10 and \
                                                 self.globalTrack().hitPattern().numberOfValidMuonHits() > 0 and \
                                                 self.numberOfMatchedStations()>1 and \
                                                 self.innerTrack().hitPattern().numberOfValidPixelHits()>0 and \
                                                 self.innerTrack().hitPattern().trackerLayersWithMeasurement() > 5
            if name == "POG_ID_Medium":
                if not self.looseId(): return False
                goodGlb = self.physObj.isGlobalMuon() and self.physObj.globalTrack().normalizedChi2() < 3 and self.physObj.combinedQuality().chi2LocalPosition < 12 and self.physObj.combinedQuality().trkKink < 20;
                return self.physObj.innerTrack().validFraction() >= 0.8 and self.physObj.segmentCompatibility() >= (0.303 if goodGlb else 0.451)
            if name == "POG_Global_OR_TMArbitrated":
                return self.physObj.isGlobalMuon() or (self.physObj.isTrackerMuon() and self.physObj.numberOfMatchedStations() > 0)
        return self.physObj.muonID(name)
            
    def mvaId(self):
        '''For a transparent treatment of electrons and muons. Returns -99'''
        return -99
    

    def dxy(self, vertex=None):
        '''either pass the vertex, or set associatedVertex before calling the function.
        note: the function does not work with standalone muons as innerTrack
        is not available.
        '''
        if vertex is None:
            vertex = self.associatedVertex
        return getattr(self,self._trackForDxyDz)().dxy( vertex.position() )

    def edxy(self):
        '''returns the uncertainty on dxy (from gsf track)'''
        return getattr(self,self._trackForDxyDz)().dxyError()
 

    def dz(self, vertex=None):
        '''either pass the vertex, or set associatedVertex before calling the function.
        note: the function does not work with standalone muons as innerTrack
        is not available.
        '''
        if vertex is None:
            vertex = self.associatedVertex
        return getattr(self,self._trackForDxyDz)().dz( vertex.position() )

    def edz(self):
        '''returns the uncertainty on dxz (from gsf track)'''
        return getattr(self,self._trackForDxyDz)().dzError()

    def chargedHadronIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumChargedHadronPt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumChargedHadronPt 
        raise RuntimeError, "Muon chargedHadronIso missing for R=%s" % R

    def neutralHadronIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumNeutralHadronEt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumNeutralHadronEt 
        raise RuntimeError, "Muon neutralHadronIso missing for R=%s" % R

    def photonIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumPhotonEt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumPhotonEt 
        raise RuntimeError, "Muon photonIso missing for R=%s" % R

    def chargedAllIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumChargedParticlePt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumChargedParticlePt 
        raise RuntimeError, "Muon chargedAllIso missing for R=%s" % R

    def chargedAllIso(self):
        return self.chargedAllIsoR()

    def puChargedHadronIsoR(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumPUPt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumPUPt 
        raise RuntimeError, "Muon chargedHadronIso missing for R=%s" % R


    def absIsoWithFSR(self, R=0.4, puCorr="deltaBeta", dBetaFactor=0.5):
        '''
        Calculate Isolation, subtract FSR, apply specific PU corrections" 
        '''
        photonIso = self.photonIsoR(R)
        if hasattr(self,'fsrPhotons'):
            for gamma in self.fsrPhotons:
                dr = deltaR(gamma.eta(), gamma.phi(), self.physObj.eta(), self.physObj.phi())
                if dr > 0.01 and dr < R:
                    photonIso = max(photonIso-gamma.pt(),0.0)                
        if puCorr == "deltaBeta":
            offset = dBetaFactor * self.puChargedHadronIsoR(R)
        elif puCorr == "rhoArea":
            offset = self.rho*getattr(self,"EffectiveArea"+(str(R).replace(".","")))
        elif puCorr in ["none","None",None]:
            offset = 0
        else:
             raise RuntimeError, "Unsupported PU correction scheme %s" % puCorr
        return self.chargedHadronIsoR(R)+max(0.,photonIso+self.neutralHadronIsoR(R)-offset)            
