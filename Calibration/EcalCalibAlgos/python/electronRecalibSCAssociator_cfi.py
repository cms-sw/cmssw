import FWCore.ParameterSet.Config as cms

electronRecalibSCAssociator = cms.EDFilter("ElectronRecalibSuperClusterAssociator",
    electronCollection = cms.string(''),
    scIslandCollection = cms.string('IslandEndcapRecalibSC'),
    scIslandProducer = cms.string('correctedIslandEndcapSuperClusters'),
    scProducer = cms.string('correctedHybridSuperClusters'),
    electronProducer = cms.string('electronFilter'),
    scCollection = cms.string('recalibSC')
)


