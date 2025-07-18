import FWCore.ParameterSet.Config as cms

hltHighPtTripletStepClusters = cms.EDProducer("TrackClusterRemoverPhase2",
    TrackQuality = cms.string('highPurity'),
    maxChi2 = cms.double(9.0),
    mightGet = cms.optional.untracked.vstring,
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    oldClusterRemovalInfo = cms.InputTag(""),
    overrideTrkQuals = cms.InputTag(""),
    phase2OTClusters = cms.InputTag("hltSiPhase2Clusters"),
    phase2pixelClusters = cms.InputTag("hltSiPixelClusters"),
    trackClassifier = cms.InputTag("","QualityMasks"),
    trajectories = cms.InputTag("hltInitialStepTrackSelectionHighPurity")
)

_hltHighPtTripletStepClustersLST = hltHighPtTripletStepClusters.clone(
    trajectories = "hltInitialStepSeedTracksLST"
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(hltHighPtTripletStepClusters, _hltHighPtTripletStepClustersLST)
