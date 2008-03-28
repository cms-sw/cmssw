import FWCore.ParameterSet.Config as cms

# Island BasicCluster producer
hltIslandBasicClusters = cms.EDProducer("EgammaHLTIslandClusterProducer",
    endcapHitProducer = cms.InputTag("ecalRecHit"),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    regionEtaMargin = cms.double(0.3), ##MARCO 0.14

    regionPhiMargin = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(1.2),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(True),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5),
    l1LowerThr = cms.double(0.0),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    l1TagIsolated = cms.InputTag("l1extraParticles","Isolated"),
    doEndcaps = cms.bool(True),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    l1TagNonIsolated = cms.InputTag("l1extraParticles","NonIsolated"),
    barrelHitProducer = cms.InputTag("ecalRecHit"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    posCalc_t0_barl = cms.double(7.4),
    doBarrel = cms.bool(True)
)


