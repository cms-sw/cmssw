import FWCore.ParameterSet.Config as cms

# Island BasicCluster producer
islandBasicClusters = cms.EDProducer("IslandClusterProducer",
	barrelHits = cms.InputTag('ecalRecHit','EcalRecHitsEB'),
	endcapHits = cms.InputTag('ecalRecHit','EcalRecHitsEE'), 			 
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    IslandEndcapSeedThr = cms.double(0.18),
    barrelShapeAssociation = cms.string('islandBarrelShapeAssoc'),
    clustershapecollectionEE = cms.string('islandEndcapShape'),
    clustershapecollectionEB = cms.string('islandBarrelShape'),
    VerbosityLevel = cms.string('ERROR'),
    endcapShapeAssociation = cms.string('islandEndcapShapeAssoc'),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5),
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(3.1),        
                                  T0_endcPresh = cms.double(1.2),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 ),
    # recHit flags to be excluded from seeding
    SeedRecHitFlagToBeExcludedEB = cms.vstring(
        'kFaultyHardware',
        'kTowerRecovered',
        'kDead'
        ),
    SeedRecHitFlagToBeExcludedEE = cms.vstring(
        'kFaultyHardware',
        'kNeighboursRecovered',
        'kTowerRecovered',
        'kDead',
        'kWeird'
        ),
    # recHit flags to be excluded from clustering
    RecHitFlagToBeExcludedEB = cms.vstring(
        'kWeird',
        'kDiWeird',
        'kOutOfTime',
        'kTowerRecovered'
        ),
    RecHitFlagToBeExcludedEE = cms.vstring(
        'kWeird',
        'kDiWeird',
        'kOutOfTime',
        'kTowerRecovered'
        )
)


