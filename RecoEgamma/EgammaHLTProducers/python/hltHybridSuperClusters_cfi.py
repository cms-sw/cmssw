import FWCore.ParameterSet.Config as cms

# Producer for Hybrid BasicClusters and SuperClusters
hltHybridSuperClusters = cms.EDProducer("EgammaHLTHybridClusterProducer",
    regionEtaMargin = cms.double(0.14),
    regionPhiMargin = cms.double(0.4),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    # position calculation parameters
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(True),
    # output collections
    #    string clustershapecollection = ""
    basicclusterCollection = cms.string(''),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    l1LowerThr = cms.double(0.0),
    eseed = cms.double(0.35),
    ethresh = cms.double(0.1),
    ewing = cms.double(1.0),
    step = cms.int32(10),
    #    string shapeAssociation = "hybridShapeAssoc"
    debugLevel = cms.string('INFO'),
    # L1 trigger candidate matching parameters
    l1TagIsolated = cms.InputTag("l1extraParticles","Isolated"),
    superclusterCollection = cms.string(''),
    posCalc_x0 = cms.double(0.89),
    # clustering parameters
    HybridBarrelSeedThr = cms.double(1.0),
    l1TagNonIsolated = cms.InputTag("l1extraParticles","NonIsolated"),
    posCalc_t0 = cms.double(7.4),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    # input collection
    ecalhitproducer = cms.InputTag("ecalRecHit")
)


