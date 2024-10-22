import FWCore.ParameterSet.Config as cms


# Multi5x5 BasicCluster producer
multi5x5BasicClustersCleaned = cms.EDProducer("Multi5x5ClusterProducer",

    # which regions should be clusterized
    doEndcap = cms.bool(True),
    doBarrel = cms.bool(False),

	barrelHitTag = cms.InputTag('ecalRecHit','EcalRecHitsEB'),
	endcapHitTag = cms.InputTag('ecalRecHit','EcalRecHitsEE'),  
    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
	endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
    IslandEndcapSeedThr = cms.double(0.18),
 
    IslandBarrelSeedThr = cms.double(0.5),
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(3.1),        
                                  T0_endcPresh = cms.double(1.2),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 ),
    reassignSeedCrysToClusterItSeeds = cms.bool(True),
    # recHit flags to be excluded from seeding

    RecHitFlagToBeExcluded = cms.vstring(
        'kFaultyHardware',
        'kNeighboursRecovered',
        'kTowerRecovered',
        'kDead',
        'kWeird',

    )
)

# with no cleaning

multi5x5BasicClustersUncleaned = multi5x5BasicClustersCleaned.clone(
    RecHitFlagToBeExcluded = []
)
