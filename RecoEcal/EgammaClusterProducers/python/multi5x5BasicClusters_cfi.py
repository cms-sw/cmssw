import FWCore.ParameterSet.Config as cms

# Multi5x5 BasicCluster producer
multi5x5BasicClusters = cms.EDProducer("Multi5x5ClusterProducer",
    posCalc_x0 = cms.double(0.89),
    endcapHitProducer = cms.string('ecalRecHit'),
    posCalc_t0_endcPresh = cms.double(1.2),
    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    barrelShapeAssociation = cms.string('multi5x5BarrelShapeAssoc'),
    posCalc_w0 = cms.double(4.2),
    posCalc_logweight = cms.bool(True),
    clustershapecollectionEE = cms.string('multi5x5EndcapShape'),
    clustershapecollectionEB = cms.string('multi5x5BarrelShape'),
    VerbosityLevel = cms.string('ERROR'),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    barrelHitProducer = cms.string('ecalRecHit'),
    endcapShapeAssociation = cms.string('multi5x5EndcapShapeAssoc'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_barl = cms.double(7.4),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5)
)


