import FWCore.ParameterSet.Config as cms


# Producer for Hybrid BasicClusters and SuperClusters
cleanedHybridSuperClusters = cms.EDProducer("HybridClusterProducer",
    eThreshA = cms.double(0.003),
    # seed thresold for dominos
    eseed = cms.double(0.35),
    # coefficient to increase Eseed as a function of 5x5; 0 gives old behaviour   
    xi = cms.double(0.00),
    # increase Eseed as a function of et_5x5 (othwewise it's e_5x5)
    useEtForXi = cms.bool(True),
    # output collections
    clustershapecollection = cms.string(''),
    shapeAssociation = cms.string('hybridShapeAssoc'),
    # if e1x3 larger than ewing use 1x5
    # e.g. always build 1x5
    ewing = cms.double(0.0),
    # clustering parameters
    #
    # threshold on seed RecHits
    HybridBarrelSeedThr = cms.double(1.0),
    dynamicPhiRoad = cms.bool(False),
    basicclusterCollection = cms.string('hybridBarrelBasicClusters'),
    # phi road parameters
    step = cms.int32(17),
    eThreshB = cms.double(0.1),
    dynamicEThresh = cms.bool(False),
    # domino thresholds
    ethresh = cms.double(0.1),
    superclusterCollection = cms.string(''),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    # input collection
    ecalhitproducer = cms.string('ecalRecHit'),
    # recHit flags to be excluded from seeding
    RecHitFlagToBeExcluded = cms.vstring(
        'kFaultyHardware',
        'kTowerRecovered',
        'kDead'
        ),
    RecHitSeverityToBeExcluded = cms.vstring('kWeird',
                                             'kBad',
                                             'kTime'),
   
    excludeFlagged = cms.bool(True),
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(3.1),        
                                  T0_endcPresh = cms.double(1.2),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 )
 ) 
