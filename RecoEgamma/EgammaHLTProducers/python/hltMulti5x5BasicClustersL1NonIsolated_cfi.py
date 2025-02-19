import FWCore.ParameterSet.Config as cms

hltMulti5x5BasicClustersL1NonIsolated = cms.EDProducer("EgammaHLTMulti5x5ClusterProducer",
    endcapHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit"),
    barrelClusterCollection = cms.string('notused'),
    regionEtaMargin = cms.double(0.3),
    regionPhiMargin = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    doIsolated = cms.bool(False),
    Multi5x5EndcapSeedThr = cms.double(0.5),
    l1UpperThr = cms.double(999.0),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
    l1LowerThr = cms.double(5.0),
    l1TagIsolated = cms.InputTag("hltL1extraParticles","Isolated"),
    doEndcaps = cms.bool(True),
    Multi5x5BarrelSeedThr = cms.double(0.5),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    l1TagNonIsolated = cms.InputTag("hltL1extraParticles","NonIsolated"),
    barrelHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    RecHitFlagToBeExcluded = cms.vstring(),
    doBarrel = cms.bool(False),
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(3.1),        
                                  T0_endcPresh = cms.double(1.2),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 )
)


