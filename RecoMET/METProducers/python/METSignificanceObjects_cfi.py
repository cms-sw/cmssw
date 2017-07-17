import FWCore.ParameterSet.Config as cms

##======================================
## Muons
##======================================

selectedMuons = cms.EDFilter(
      "MuonSelector",
      src = cms.InputTag('muons'),
      cut = cms.string(    "(isTrackerMuon) && pt > 5"
         "&& isPFMuon"+
         "&& globalTrack.isNonnull"+
         "&& innerTrack.hitPattern.numberOfValidPixelHits > 0"+
         "&& innerTrack.normalizedChi2 < 10"+
         "&& numberOfMatches > 0"+
         "&& innerTrack.hitPattern.numberOfValidTrackerHits>5"+
         "&& globalTrack.hitPattern.numberOfValidHits>0"+
         "&& (pfIsolationR03.sumChargedHadronPt+max(0.,pfIsolationR03.sumNeutralHadronEt+pfIsolationR03.sumPhotonEt - 0.5*pfIsolationR03.sumPUPt))/pt < 0.3"+
         "&& abs(innerTrack().dxy)<2.0"
         ),
      filter = cms.bool(False)
      )


##======================================
## Electrons
##======================================

selectedElectrons = cms.EDFilter(
      "GsfElectronSelector",
      src = cms.InputTag('gedGsfElectrons'),
      cut = cms.string(
         "abs(eta) < 2.5 && pt > 19.5"                              +
         "&& (gsfTrack.hitPattern().numberOfHits(\'MISSING_INNER_HITS\')<=1 )" +
         "&& (pfIsolationVariables.sumChargedHadronPt+max(0.,pfIsolationVariables.sumNeutralHadronEt+pfIsolationVariables.sumPhotonEt - 0.5*pfIsolationVariables.sumPUPt))/et     < 0.3"  +
         "&& ((abs(eta) < 1.4442  "                                 +
         "&& abs(deltaEtaSuperClusterTrackAtVtx)            < 0.007"+
         "&& abs(deltaPhiSuperClusterTrackAtVtx)            < 0.8"  +
         "&& sigmaIetaIeta                                  < 0.01" +
         "&& hcalOverEcal                                   < 0.15" +
         "&& abs(1./superCluster.energy - 1./p)             < 0.05)"+
         "|| (abs(eta)  > 1.566 "+
         "&& abs(deltaEtaSuperClusterTrackAtVtx)            < 0.009"+
         "&& abs(deltaPhiSuperClusterTrackAtVtx)            < 0.10" +
         "&& sigmaIetaIeta                                  < 0.03" +
         "&& hcalOverEcal                                   < 0.10" +
         "&& abs(1./superCluster.energy - 1./p)             < 0.05))" 
         ),
      filter = cms.bool(False)
      )


##======================================
## Photons
##======================================

selectedPhotons = cms.EDFilter("PhotonSelector",
      src = cms.InputTag("photons"),
      cut = cms.string(
         "abs(eta) < 2.5 && pt > 19.5" +
         "&& sigmaIetaIeta < 0.03" +
         "&& hadronicOverEm < 0.05" +
         "&& hasPixelSeed == 0" +
         "&& (chargedHadronIso + neutralHadronIso + photonIso)/pt < 0.2"
         )
      )


selectionSequenceForMETSig = cms.Sequence(
      selectedMuons+
      selectedElectrons+
      selectedPhotons
      )
