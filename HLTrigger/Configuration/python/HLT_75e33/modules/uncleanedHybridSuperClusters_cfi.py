import FWCore.ParameterSet.Config as cms

uncleanedHybridSuperClusters = cms.EDProducer("HybridClusterProducer",
    HybridBarrelSeedThr = cms.double(1.0),
    RecHitFlagToBeExcluded = cms.vstring(
        'kFaultyHardware',
        'kTowerRecovered',
        'kDead'
    ),
    RecHitSeverityToBeExcluded = cms.vstring(),
    basicclusterCollection = cms.string('hybridBarrelBasicClusters'),
    clustershapecollection = cms.string(''),
    dynamicEThresh = cms.bool(False),
    dynamicPhiRoad = cms.bool(False),
    eThreshA = cms.double(0.003),
    eThreshB = cms.double(0.1),
    eseed = cms.double(0.35),
    ethresh = cms.double(0.1),
    ewing = cms.double(0.0),
    excludeFlagged = cms.bool(False),
    posCalcParameters = cms.PSet(
        LogWeighted = cms.bool(True),
        T0_barl = cms.double(7.4),
        T0_endc = cms.double(3.1),
        T0_endcPresh = cms.double(1.2),
        W0 = cms.double(4.2),
        X0 = cms.double(0.89)
    ),
    recHitsCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    shapeAssociation = cms.string('hybridShapeAssoc'),
    step = cms.int32(17),
    superclusterCollection = cms.string(''),
    useEtForXi = cms.bool(True),
    xi = cms.double(0.0)
)
