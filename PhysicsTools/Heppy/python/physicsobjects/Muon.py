from PhysicsTools.Heppy.physicsobjects.Lepton import Lepton

class Muon( Lepton ):

    def looseId( self ):
        '''Loose ID as recommended by mu POG.'''
        return self.physObj.isLooseMuon()

    def tightId( self ):
        '''Tight ID as recommended by mu POG.'''
        return self.muonID("POG_ID_Tight")

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
        return self.physObj.muonID(name)
            
    def mvaId(self):
        '''For a transparent treatment of electrons and muons. Returns -99'''
        return -99
    
   

    def absEffAreaIso(self,rho,effectiveAreas):
        return self.absIsoFromEA(rho,self.eta(),effectiveAreas.muon)



    def dxy(self, vertex=None):
        '''either pass the vertex, or set associatedVertex before calling the function.
        note: the function does not work with standalone muons as innerTrack
        is not available.
        '''
        if vertex is None:
            vertex = self.associatedVertex
        return self.innerTrack().dxy( vertex.position() )
 

    def dz(self, vertex=None):
        '''either pass the vertex, or set associatedVertex before calling the function.
        note: the function does not work with standalone muons as innerTrack
        is not available.
        '''
        if vertex is None:
            vertex = self.associatedVertex
        return self.innerTrack().dz( vertex.position() )

    def chargedHadronIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumChargedHadronPt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumChargedHadronPt 
        raise RuntimeError, "Muon chargedHadronIso missing for R=%s" % R

    def neutralHadronIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumNeutralHadronEt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumNeutralHadronEt 
        raise RuntimeError, "Muon neutralHadronIso missing for R=%s" % R

    def photonIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumPhotonEt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumPhotonEt 
        raise RuntimeError, "Muon photonIso missing for R=%s" % R

    def chargedAllIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumChargedParticlePt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumChargedParticlePt 
        raise RuntimeError, "Muon chargedAllIso missing for R=%s" % R

    def puChargedHadronIso(self,R=0.4):
        if   R == 0.3: return self.physObj.pfIsolationR03().sumPUPt 
        elif R == 0.4: return self.physObj.pfIsolationR04().sumPUPt 
        raise RuntimeError, "Muon chargedHadronIso missing for R=%s" % R




