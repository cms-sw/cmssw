import FWCore.ParameterSet.Config as cms

siStripSeeds = cms.EDProducer("ElectronSiStripSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
)


