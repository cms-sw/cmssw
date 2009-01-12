import FWCore.ParameterSet.Config as cms

siStripSeeds = cms.EDProducer("SiStripElectronSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
)


