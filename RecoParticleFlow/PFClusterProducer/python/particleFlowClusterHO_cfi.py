import FWCore.ParameterSet.Config as cms

# Use this object to modify parameters specifically for Run 2

#### PF CLUSTER HO ####

#cleaning

#seeding
_localMaxSeeds_HO = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
              seedingThreshold = cms.double(1.0),
              seedingThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
              seedingThreshold = cms.double(3.1),
              seedingThresholdPt = cms.double(0.0)
              )
    ),
    nNeighbours = cms.int32(4)
)

#topo clusters
_topoClusterizer_HO = cms.PSet(
    algoName = cms.string("Basic2DGenericTopoClusterizer"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
              gatheringThreshold = cms.double(0.5),
              gatheringThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
              gatheringThreshold = cms.double(1.0),
              gatheringThresholdPt = cms.double(0.0)
              )
    ),
    useCornerCells = cms.bool(True)
)

#position calc
_positionCalcHO_cross_nodepth = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(5),
    logWeightDenominator = cms.double(0.5), # same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)    
)

_positionCalcHO_all_nodepth = _positionCalcHO_cross_nodepth.clone(
    posCalcNCrystals = cms.int32(-1)
)

#pf clusters
_pfClusterizer_HO = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = _positionCalcHO_cross_nodepth,
    allCellsPositionCalc = _positionCalcHO_all_nodepth,
    showerSigma = cms.double(10.0),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
              recHitEnergyNorm = cms.double(0.5)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
              recHitEnergyNorm = cms.double(1.0)
              )
    )
)

particleFlowClusterHO = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHO"),
    recHitCleaners = cms.VPSet(),
    seedCleaners  = cms.VPSet(),
    seedFinder = _localMaxSeeds_HO,
    initialClusteringStep = _topoClusterizer_HO,
    pfClusterBuilder = _pfClusterizer_HO,
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

#
# Need to change the quality tests for Run 2
#
def _modifyParticleFlowClusterHOForRun2( object ) :
    """
    Customises PFClusterProducer for Run 2.
    """
    for p in object.seedFinder.thresholdsByDetector:
        p.seedingThreshold = cms.double(0.08)

    for p in object.initialClusteringStep.thresholdsByDetector:
        p.gatheringThreshold = cms.double(0.05)

    for p in object.pfClusterBuilder.recHitEnergyNorms:
        p.recHitEnergyNorm = cms.double(0.05)

    object.pfClusterBuilder.positionCalc.logWeightDenominator = cms.double(0.05)
    object.pfClusterBuilder.allCellsPositionCalc.logWeightDenominator = cms.double(0.05)

# Call the function above to modify particleFlowClusterHO only if the run2 era is active
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( particleFlowClusterHO, func=_modifyParticleFlowClusterHOForRun2 )
