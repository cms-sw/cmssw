import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.ecalRecHitFlags_cfi import *
from RecoEcal.EgammaClusterProducers.ecalSeverityLevelAlgos_cfi import *
from RecoEcal.EgammaClusterProducers.ecalSeverityLevelFlags_cfi import *

# Producer for Hybrid BasicClusters and SuperClusters
cleanedHybridSuperClusters = cms.EDProducer("HybridClusterProducer",
    eThreshA = cms.double(0.003),
    # seed thresold for dominos
    eseed = cms.double(0.35),
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
    debugLevel = cms.string('INFO'),
    dynamicEThresh = cms.bool(False),
    # domino thresholds
    ethresh = cms.double(0.1),
    superclusterCollection = cms.string(''),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    # input collection
    ecalhitproducer = cms.string('ecalRecHit'),
    # recHit flags to be excluded from seeding
    RecHitFlagToBeExcluded = cms.vint32(
        ecalRecHitFlag_kFaultyHardware,
        ecalRecHitFlag_kPoorCalib,
        #        ecalRecHitFlag_kSaturated,
        #        ecalRecHitFlag_kLeadingEdgeRecovered,
        #        ecalRecHitFlag_kNeighboursRecovered,
        ecalRecHitFlag_kTowerRecovered,
        ecalRecHitFlag_kDead
        ),
    RecHitSeverityToBeExcluded = cms.vint32(ecalSeverityLevelFlag_kWeird,ecalSeverityLevelFlag_kBad,ecalSeverityLevelFlag_kTime),
    severityRecHitThreshold = cms.double(4.),
    severitySpikeId = cms.int32(ecalSeverityLevelSpikeId_kSwissCrossBordersIncluded),
    severitySpikeThreshold = cms.double(0.95),
    excludeFlagged = cms.bool(True),
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(3.1),        
                                  T0_endcPresh = cms.double(1.2),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 )
 ) 
