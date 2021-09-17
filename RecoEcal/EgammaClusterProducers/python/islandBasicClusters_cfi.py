import FWCore.ParameterSet.Config as cms
import RecoEcal.EgammaClusterProducers.IslandClusterProducer_cfi as _mod

# Island BasicCluster producer
islandBasicClusters = _mod.IslandClusterProducer.clone(
    endcapShapeAssociation = 'islandEndcapShapeAssoc',
    posCalcParameters = dict(),
    # recHit flags to be excluded from seeding
    SeedRecHitFlagToBeExcludedEB = [
        'kFaultyHardware',
        'kTowerRecovered',
        'kDead'
        ],
    SeedRecHitFlagToBeExcludedEE = [
        'kFaultyHardware',
        'kNeighboursRecovered',
        'kTowerRecovered',
        'kDead',
        'kWeird'
        ],
    # recHit flags to be excluded from clustering
    RecHitFlagToBeExcludedEB = [
        'kWeird',
        'kDiWeird',
        'kOutOfTime',
        'kTowerRecovered'
        ],
    RecHitFlagToBeExcludedEE = [
        'kWeird',
        'kDiWeird',
        'kOutOfTime',
        'kTowerRecovered'
        ]
)
