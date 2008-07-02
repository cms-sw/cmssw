import FWCore.ParameterSet.Config as cms

hltMulti5x5BasicClustersL1Isolated = cms.EDProducer("EgammaHLTMulti5x5ClusterProducer",
    endcapHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit"),
    barrelClusterCollection = cms.string('notused'),
    regionEtaMargin = cms.double(0.3),
    regionPhiMargin = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(1.2),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(True),
    Multi5x5EndcapSeedThr = cms.double(0.5),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
    l1LowerThr = cms.double(5.0),
    posCalc_t0_endc = cms.double(3.1),
    l1TagIsolated = cms.InputTag("hltL1extraParticles","Isolated"),
    doEndcaps = cms.bool(True),
    Multi5x5BarrelSeedThr = cms.double(0.5),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    l1TagNonIsolated = cms.InputTag("hltL1extraParticles","NonIsolated"),
    barrelHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    posCalc_t0_barl = cms.double(7.4),
    doBarrel = cms.bool(False)
)


