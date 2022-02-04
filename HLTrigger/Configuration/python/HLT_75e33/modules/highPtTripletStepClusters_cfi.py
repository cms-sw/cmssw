import FWCore.ParameterSet.Config as cms

highPtTripletStepClusters = cms.EDProducer("TrackClusterRemoverPhase2",
    TrackQuality = cms.string('highPurity'),
    maxChi2 = cms.double(9.0),
    mightGet = cms.optional.untracked.vstring,
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    oldClusterRemovalInfo = cms.InputTag(""),
    overrideTrkQuals = cms.InputTag(""),
    phase2OTClusters = cms.InputTag("siPhase2Clusters"),
    phase2pixelClusters = cms.InputTag("siPixelClusters"),
    trackClassifier = cms.InputTag("","QualityMasks"),
    trajectories = cms.InputTag("initialStepTrackSelectionHighPurity")
)
