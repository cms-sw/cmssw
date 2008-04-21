import FWCore.ParameterSet.Config as cms

trackAlgoCompare = cms.EDFilter("TrackAlgoCompareUtil",
    vertexLabel_algoB = cms.InputTag("offlinePrimaryVertices"),
    #InputTag trackingParticleLabel_fakes = cutsTPFake
    #InputTag trackingParticleLabel_effic = cutsTPEffic
    vertexLabel_algoA = cms.InputTag("offlinePrimaryVerticesFromCTFTracks"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    #InputTag vertexLabel_algoB = offlinePrimaryVerticesFromRSTracks
    trackingVertexLabel = cms.InputTag("mergedtruth","MergedTrackTruth"),
    trackLabel_algoB = cms.InputTag("rsWithMaterialTracks"),
    trackLabel_algoA = cms.InputTag("generalTracks"),
    assocLabel = cms.untracked.string('TrackAssociatorByHits'),
    trackingParticleLabel_effic = cms.InputTag("mergedtruth","MergedTrackTruth"),
    trackingParticleLabel_fakes = cms.InputTag("mergedtruth","MergedTrackTruth")
)


