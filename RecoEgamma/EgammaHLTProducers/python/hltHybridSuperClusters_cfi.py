import FWCore.ParameterSet.Config as cms


# Producer for Hybrid BasicClusters and SuperClusters
hltHybridSuperClusters = cms.EDProducer("EgammaHLTHybridClusterProducer",
    regionEtaMargin = cms.double(0.14),
    regionPhiMargin = cms.double(0.4),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    # position calculation parameters
    doIsolated = cms.bool(True),
    # output collections
    #    string clustershapecollection = ""
    basicclusterCollection = cms.string(''),
    l1UpperThr = cms.double(999.0),
    l1LowerThr = cms.double(0.0),
    eseed = cms.double(0.35),
    # coefficient to increase Eseed\ as a function of 5x5; 0 gives old behaviour   
    xi = cms.double(0.00),
    # increase Eseed as a function of et_5x5 (othwewise it's e_5x5)
    useEtForXi = cms.bool(True),
    ethresh = cms.double(0.1),
    ewing = cms.double(1.0),
    step = cms.int32(10),
    #    string shapeAssociation = "hybridShapeAssoc"
    debugLevel = cms.string('INFO'),
    # L1 trigger candidate matching parameters
    l1TagIsolated = cms.InputTag("l1extraParticles","Isolated"),
    superclusterCollection = cms.string(''),
    # clustering parameters
    HybridBarrelSeedThr = cms.double(1.0),
    l1TagNonIsolated = cms.InputTag("l1extraParticles","NonIsolated"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    # flags to be excluded
    RecHitFlagToBeExcluded = cms.vstring(),
    # new spikeId removal. Off by default
    RecHitSeverityToBeExcluded = cms.vstring(),
    severityRecHitThreshold = cms.double(4.),
    excludeFlagged = cms.bool(False),
    eThreshB = cms.double(0.1),
    eThreshA = cms.double(0.003),                                        
    dynamicEThresh = cms.bool(False),
    dynamicPhiRoad = cms.bool(False),
    # input collection
    ecalhitproducer = cms.InputTag("ecalRecHit"),
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(6.3),        
                                  T0_endcPresh = cms.double(3.6),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 ),                                            
)


