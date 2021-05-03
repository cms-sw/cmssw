import FWCore.ParameterSet.Config as cms

hltElePixelHitTripletsClusterRemoverUnseeded = cms.EDProducer("SeedClusterRemoverPhase2",
    phase2OTClusters = cms.InputTag("siPhase2Clusters"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    trajectories = cms.InputTag("hltElePixelSeedsTripletsUnseeded")
)
