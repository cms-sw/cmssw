import FWCore.ParameterSet.Config as cms

electronRecalibSCAssociator = cms.EDProducer("ElectronRecalibSuperClusterAssociator",
    electronCollection = cms.string(''),
    scIslandCollection = cms.string('IslandEndcapRecalibSC'),
    scIslandProducer = cms.string('correctedIslandEndcapSuperClusters'),
    scProducer = cms.string('correctedHybridSuperClusters'),
    electronProducer = cms.string('electronFilter'),
    scCollection = cms.string('recalibSC')
)


