import FWCore.ParameterSet.Config as cms

#IsolatedPixelTrackCandidateProducer default configuration
isolPixelTrackProd = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
    L1GtObjectMapSource = cms.InputTag("l1GtEmulDigis"),
    L1eTauJetsSource = cms.InputTag("l1extraParticles","Tau"),
    tauAssociationCone = cms.double(0.5),
    PixelTracksSource = cms.InputTag("pixelTracks"),
    L1GTSeedLabel = cms.InputTag("l1sIsolTrack"),
    tauUnbiasCone = cms.double(0.0),
    PixelIsolationConeSize = cms.double(0.2),
    ecalFilterLabel = cms.InputTag("aaa")
)


