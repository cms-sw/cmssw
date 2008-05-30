import FWCore.ParameterSet.Config as cms

trackAlgoCompare = cms.EDFilter("TrackAlgoCompareUtil",

    trackLabel_algoA = cms.InputTag("generalTracks"),
    trackLabel_algoB = cms.InputTag("rsWithMaterialTracks"),
    vertexLabel_algoA = cms.InputTag("offlinePrimaryVerticesFromCTFTracks"),
    vertexLabel_algoB = cms.InputTag("offlinePrimaryVerticesFromCTFTracks"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    
    trackingVertexLabel = cms.InputTag("mergedtruth","MergedTrackTruth"),
    trackingParticleLabel_effic = cms.InputTag("mergedtruth","MergedTrackTruth"),
    trackingParticleLabel_fakes = cms.InputTag("mergedtruth","MergedTrackTruth"),

    assocLabel = cms.untracked.string('TrackAssociatorByHits'),
    UseAssociators = cms.bool("true"),
    associatormap_algoA = cms.InputTag("trackingParticleRecoTrackAsssociation"),
    associatormap_algoB = cms.InputTag("assoc2secStepTk")
)


