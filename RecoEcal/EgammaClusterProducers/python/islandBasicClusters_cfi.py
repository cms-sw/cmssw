import FWCore.ParameterSet.Config as cms

# Island BasicCluster producer
islandBasicClusters = cms.EDProducer("IslandClusterProducer",
    endcapHitProducer = cms.string('ecalRecHit'),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    IslandEndcapSeedThr = cms.double(0.18),
    barrelShapeAssociation = cms.string('islandBarrelShapeAssoc'),
    clustershapecollectionEE = cms.string('islandEndcapShape'),
    clustershapecollectionEB = cms.string('islandBarrelShape'),
    VerbosityLevel = cms.string('ERROR'),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    barrelHitProducer = cms.string('ecalRecHit'),
    endcapShapeAssociation = cms.string('islandEndcapShapeAssoc'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5),
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(3.1),        
                                  T0_endcPresh = cms.double(1.2),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 )
)


