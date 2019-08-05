import FWCore.ParameterSet.Config as cms

selectDigi = cms.EDProducer("EcalDigiSelector",
    EcalEBDigiTag = cms.InputTag("ecalDigis","ebDigis"),
    nclus_sel = cms.int32(2),
    EcalEEDigiTag = cms.InputTag("ecalDigis","eeDigis"),
    barrelSuperClusterProducer = cms.InputTag('uncleanedHybridSuperClusters'),
    endcapSuperClusterProducer = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'),
    EcalEBRecHitTag = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EcalEERecHitTag = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    cluster_pt_thresh = cms.double(10.0),
    single_cluster_thresh = cms.double(15.0),
    selectedEcalEBDigiCollection = cms.string('selectedEcalEBDigiCollection'),
    selectedEcalEEDigiCollection = cms.string('selectedEcalEEDigiCollection')
)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(selectDigi,
                           cluster_pt_thresh = 1.0,
                           single_cluster_thresh = 1.0)
