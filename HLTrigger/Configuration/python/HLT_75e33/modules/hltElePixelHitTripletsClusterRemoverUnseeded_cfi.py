import FWCore.ParameterSet.Config as cms

hltElePixelHitTripletsClusterRemoverUnseeded = cms.EDProducer("SeedClusterRemoverPhase2",
    phase2OTClusters = cms.InputTag("hltSiPhase2Clusters"),
    pixelClusters = cms.InputTag("hltSiPixelClusters"),
    trajectories = cms.InputTag("hltElePixelSeedsTripletsUnseeded")
)
