import FWCore.ParameterSet.Config as cms

#IsolatedPixelTrackCandidateProducer default configuration
isolPixelTrackProd = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
    L1eTauJetsSource = cms.InputTag("l1extraParticles","Tau"),
    tauAssociationCone = cms.untracked.double(0.5),
    PixelTracksSource = cms.InputTag("pixelTracks"),
    L1GTSeedLabel = cms.InputTag("l1sIsolTrack"),
    tauUnbiasCone = cms.untracked.double(1.2),
    PixelIsolationConeSize = cms.untracked.double(0.5),
    ecalFilterLabel = cms.InputTag("aaa")
)


