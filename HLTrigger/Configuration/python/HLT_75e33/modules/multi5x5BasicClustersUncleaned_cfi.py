import FWCore.ParameterSet.Config as cms

multi5x5BasicClustersUncleaned = cms.EDProducer("Multi5x5ClusterProducer",
    IslandBarrelSeedThr = cms.double(0.5),
    IslandEndcapSeedThr = cms.double(0.18),
    RecHitFlagToBeExcluded = cms.vstring(),
    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
    barrelHitTag = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    doBarrel = cms.bool(False),
    doEndcap = cms.bool(True),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
    endcapHitTag = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    posCalcParameters = cms.PSet(
        LogWeighted = cms.bool(True),
        T0_barl = cms.double(7.4),
        T0_endc = cms.double(3.1),
        T0_endcPresh = cms.double(1.2),
        W0 = cms.double(4.2),
        X0 = cms.double(0.89)
    ),
    reassignSeedCrysToClusterItSeeds = cms.bool(True)
)
