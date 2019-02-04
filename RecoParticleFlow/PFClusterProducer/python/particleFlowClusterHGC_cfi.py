import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowRealisticSimClusterHGCCalibrations_cfi import *
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import *
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
    maxDforTimingSquared = cms.double(4.0),
    timeOffset = hgceeDigitizer.tofDelay,
    minNHitsforTiming = cms.uint32(3),
    useMCFractionsForExclEnergy = cms.bool(False),
    thresholdsByDetector = cms.VPSet(
    ),
    hadronCalib = hadronCorrections.value,
    egammaCalib = egammaCorrections.value,
    calibMinEta = minEtaCorrection,
    calibMaxEta = maxEtaCorrection,
    simClusterSrc = cms.InputTag("mix:MergedCaloTruth")
)
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(_simClusterMapper_HGCal, simClusterSrc = "mixData:MergedCaloTruth")


#position calculations
_positionCalcPCA_HGCal = cms.PSet(
        algoName = cms.string("Cluster3DPCACalculator"),
        minFractionInCalc = cms.double(1e-9),
        updateTiming = cms.bool(False)
)

_hgcalMultiClusterMapper_HGCal = cms.PSet(
    algoName = cms.string("PFClusterFromHGCalMultiCluster"),
    thresholdsByDetector = cms.VPSet(
    ),
    clusterSrc = cms.InputTag("hgcalMultiClusters")
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
