import FWCore.ParameterSet.Config as cms

selectDigi = cms.EDProducer("EcalDigiSelector",
    EcalEBDigiTag = cms.InputTag("ecalDigis","ebDigis"),
    barrelSuperClusterCollection = cms.string(''),
    nclus_sel = cms.int32(2),
    EcalEEDigiTag = cms.InputTag("ecalDigis","eeDigis"),
    barrelSuperClusterProducer = cms.string('correctedHybridSuperClusters'),
    endcapSuperClusterProducer = cms.string('correctedMulti5x5SuperClustersWithPreshower'),
    endcapSuperClusterCollection = cms.string(''),
    selectedEcalEBDigiCollection = cms.string('selectedEcalEBDigiCollection'),
    EcalEBRecHitTag = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EcalEERecHitTag = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    cluster_pt_thresh = cms.double(10.0),
    single_cluster_thresh = cms.double(15.0),
    selectedEcalEEDigiCollection = cms.string('selectedEcalEEDigiCollection')
)
