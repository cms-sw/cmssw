import FWCore.ParameterSet.Config as cms

trackAlgoCompare = cms.EDFilter("TrackAlgoCompareUtil",
    associatormap_algoA = cms.InputTag("trackingParticleRecoTrackAsssociation"),
    vertexLabel_algoB = cms.InputTag("offlinePrimaryVerticesFromCTFTracks"),
    vertexLabel_algoA = cms.InputTag("offlinePrimaryVerticesFromCTFTracks"),
    associatormap_algoB = cms.InputTag("assoc2secStepTk"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    UseAssociators = cms.bool(False),
    trackLabel_algoB = cms.InputTag("secStep"),
    trackLabel_algoA = cms.InputTag("generalTracks"),
    assocLabel = cms.untracked.string('TrackAssociatorByHits'),
    trackingParticleLabel_effic = cms.InputTag("mergedtruth","MergedTrackTruth"),
    trackingParticleLabel_fakes = cms.InputTag("mergedtruth","MergedTrackTruth")
)



