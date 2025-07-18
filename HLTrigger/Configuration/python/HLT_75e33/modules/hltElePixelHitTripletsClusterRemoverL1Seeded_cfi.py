import FWCore.ParameterSet.Config as cms

hltElePixelHitTripletsClusterRemoverL1Seeded = cms.EDProducer("SeedClusterRemoverPhase2",
    phase2OTClusters = cms.InputTag("hltSiPhase2Clusters"),
    pixelClusters = cms.InputTag("hltSiPixelClusters"),
    trajectories = cms.InputTag("hltElePixelSeedsTripletsL1Seeded")
)
