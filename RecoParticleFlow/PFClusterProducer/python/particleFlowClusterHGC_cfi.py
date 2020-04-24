import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowRealisticSimClusterHGCCalibrations_cfi import *
#### PF CLUSTER HGCal ####

#cleaning (none for now)

#seeding
_passThruSeeds_HGCal = cms.PSet(
    algoName = cms.string("PassThruSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    ),
    nNeighbours = cms.int32(8)
)

# initial step clusterizer
_simClusterMapper_HGCal = cms.PSet(
    algoName = cms.string("RealisticSimClusterMapper"),
    exclusiveFraction = cms.double(0.6),
    invisibleFraction = cms.double(0.6),
    maxDistanceFilter = cms.bool(True),
    #filtering out hits outside a cylinder of 10cm radius, built around the center of gravity per each layer
    maxDistance =  cms.double(10.0),
    useMCFractionsForExclEnergy = cms.bool(False),    
    thresholdsByDetector = cms.VPSet(
    ),
    hadronCalib = hadronCorrections,
    egammaCalib = egammaCorrections, 
    calibMinEta = minEtaCorrection, 
    calibMaxEta = maxEtaCorrection,                                   
    simClusterSrc = cms.InputTag("mix:MergedCaloTruth")
)


#position calculations
_positionCalcPCA_HGCal = cms.PSet(
        algoName = cms.string("Cluster3DPCACalculator"),
        minFractionInCalc = cms.double(1e-9)
)

_hgcalMultiClusterMapper_HGCal = cms.PSet(
    algoName = cms.string("PFClusterFromHGCalMultiCluster"),
    thresholdsByDetector = cms.VPSet(
    ),
    clusterSrc = cms.InputTag("hgcalLayerClusters")
)

particleFlowClusterHGCal = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGC"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _passThruSeeds_HGCal,
    initialClusteringStep = _simClusterMapper_HGCal,
    pfClusterBuilder = cms.PSet(),
    positionReCalc = _positionCalcPCA_HGCal,
    energyCorrector = cms.PSet()
    )

particleFlowClusterHGCalFromMultiCl = particleFlowClusterHGCal.clone(
    initialClusteringStep = _hgcalMultiClusterMapper_HGCal
)
