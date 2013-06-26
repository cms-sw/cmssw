import FWCore.ParameterSet.Config as cms

# Island BasicCluster producer
hltIslandBasicClusters = cms.EDProducer("EgammaHLTIslandClusterProducer",
    endcapHitProducer = cms.InputTag("ecalRecHit"),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    regionEtaMargin = cms.double(0.3), ##MARCO 0.14

    regionPhiMargin = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(True),
    l1UpperThr = cms.double(999.0),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5),
    l1LowerThr = cms.double(0.0),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    l1TagIsolated = cms.InputTag("l1extraParticles","Isolated"),
    doEndcaps = cms.bool(True),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    l1TagNonIsolated = cms.InputTag("l1extraParticles","NonIsolated"),
    barrelHitProducer = cms.InputTag("ecalRecHit"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    doBarrel = cms.bool(True),
                                        
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(6.3),        
                                  T0_endcPresh = cms.double(1.2),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 ),                                         
)


