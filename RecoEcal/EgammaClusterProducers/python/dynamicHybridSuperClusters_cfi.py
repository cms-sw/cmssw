import FWCore.ParameterSet.Config as cms

# Producer for Hybrid BasicClusters and SuperClusters
dynamicHybridSuperClusters = cms.EDProducer("HybridClusterProducer",
    eThreshA = cms.double(0.003),
    # seed thresold for dominos
    eseed = cms.double(0.35),
    posCalc_x0 = cms.double(0.89),
    # output collections
    clustershapecollection = cms.string(''),
    shapeAssociation = cms.string('dynamicHybridShapeAssoc'),
    # if e1x3 larger than ewing use 1x5
    # ewing = 0 so always use 1x5
    ewing = cms.double(0.0),
    # clustering parameters
    # threshold on seed RecHits
    HybridBarrelSeedThr = cms.double(1.0),
    dynamicPhiRoad = cms.bool(True),
    basicclusterCollection = cms.string(''),
    posCalc_w0 = cms.double(4.2),
    posCalc_logweight = cms.bool(True),
    # phi road parameters
    # fixed road not used as dynamicPhiRoad
    step = cms.int32(17),
    eThreshB = cms.double(0.1),
    posCalc_t0 = cms.double(7.4),
    debugLevel = cms.string('INFO'),
    dynamicEThresh = cms.bool(True),
    # domino thresholds
    # fixed ethresh not used as dynamicEThresh
    ethresh = cms.double(0.1),
    superclusterCollection = cms.string(''),
    # input collection
    ecalhitproducer = cms.string('ecalRecHit'),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    # for brem recovery
    bremRecoveryPset = cms.PSet(
        barrel = cms.PSet(
            cryVec = cms.vint32(17, 15, 13, 12, 11, 10, 9, 8, 7, 6),
            cryMin = cms.int32(5),
            etVec = cms.vdouble(5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 45.0, 135.0, 195.0, 225.0)
        ),
        endcap = cms.PSet(
            a = cms.double(47.85),
            c = cms.double(0.1201),
            b = cms.double(108.8)
        )
    )
)


