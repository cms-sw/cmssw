import FWCore.ParameterSet.Config as cms

#  BasicCluster producer
cosmicBasicClusters = cms.EDProducer("CosmicClusterProducer",
    endcapHitProducer = cms.string('ecalRecHit'),
    barrelClusterCollection = cms.string('CosmicBarrelBasicClusters'),
    EndcapSecondThr = cms.double(0.044),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(1.2),
    posCalc_logweight = cms.bool(True),
    BarrelSingleThr = cms.double(0.12221),
    barrelShapeAssociation = cms.string('CosmicBarrelShapeAssoc'),
    posCalc_w0 = cms.double(4.2),
    clustershapecollectionEE = cms.string('CosmicEndcapShape'),
    clustershapecollectionEB = cms.string('CosmicBarrelShape'),
    EndcapSingleThr = cms.double(0.135),
    maskedChannels = cms.untracked.vint32(),
    endcapClusterCollection = cms.string('CosmicEndcapBasicClusters'),
    BarrelSecondThr = cms.double(0.04073),
    EndcapSeedThr = cms.double(0.044),
    posCalc_t0_endc = cms.double(3.1),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    BarrelSeedThr = cms.double(0.04073),
    endcapShapeAssociation = cms.string('CosmicEndcapShapeAssoc'),
    barrelHitProducer = cms.string('ecalRecHit'),
    posCalc_t0_barl = cms.double(7.4)
)
