import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HCAL ####

#cleaning 
_rbxAndHPDCleaner = cms.PSet(    
    algoName = cms.string("RBXAndHPDCleaner")
)

#seeding
_localMaxSeeds_HCAL = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL1"),
              seedingThreshold = cms.double(0.5),
              seedingThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HCAL_ENDCAP"),
              seedingThreshold = cms.double(0.6),
              seedingThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
              seedingThreshold = cms.double(0.25),
              seedingThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
              seedingThreshold = cms.double(0.25),
              seedingThresholdPt = cms.double(0.0)
              )    
    ),
    nNeighbours = cms.int32(4)
)

_positionCalcHCAL_semi3D = cms.PSet(
    algoName = cms.string("Semi3DPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),    
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(0.05),#same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)
)

_positionCalcHCAL_semi3D_seedneighbours = cms.PSet(
    algoName = cms.string("Semi3DPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),    
    posCalcNCrystals = cms.int32(1), # != -1 means use the seed's neighbours
    logWeightDenominator = cms.double(0.05),#same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)
)

#topo clusters
# two possibilities "SharedSeedsClusterizer" and "ArborLikeClusterizer"
# both give similar results in QCD 80-120 and have the same handles
# strategies are outlined in the code
_topoClusterizer_HCAL = cms.PSet(
    algoName = cms.string("ArborLikeClusterizer"),    
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL1"),
              gatheringThreshold = cms.double(0.2),
              gatheringThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HCAL_ENDCAP"),
              gatheringThreshold = cms.double(0.2),
              gatheringThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
              gatheringThreshold = cms.double(0.1),
              gatheringThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
              gatheringThreshold = cms.double(0.1),
              gatheringThresholdPt = cms.double(0.0)
              )
    ),
    useCornerCells = cms.bool(True),
    showerSigma = cms.double(10.0),
    stoppingTolerance = cms.double(1e-8),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    maxIterations = cms.uint32(50),
    positionCalc = _positionCalcHCAL_semi3D_seedneighbours,
    allCellsPositionCalc = _positionCalcHCAL_semi3D
)

#position calc
_positionCalcHCAL_cross_nodepth = cms.PSet(
    algoName = cms.string("Semi3DPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),    
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(0.05),#same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)
)

_positionCalcHCAL_all_nodepth = _positionCalcHCAL_cross_nodepth.clone(
    posCalcNCrystals = cms.int32(-1)
)

#pf clusterizer
_pfClusterizer_HCAL = cms.PSet(
    algoName = cms.string("Semi3DPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = _positionCalcHCAL_cross_nodepth,
    allCellsPositionCalc = _positionCalcHCAL_all_nodepth,
    showerSigma = cms.double(10.0),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
    cms.PSet( detector = cms.string("HCAL_BARREL1"),
              recHitEnergyNorm = cms.double(0.8)
              ),
    cms.PSet( detector = cms.string("HCAL_ENDCAP"),
              recHitEnergyNorm = cms.double(0.8)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
              recHitEnergyNorm = cms.double(0.5)
              ),
    cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
              recHitEnergyNorm = cms.double(1.0)
              )
    )
)

particleFlowClusterHCALSemi3D = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHBHEHO"),
    recHitCleaners = cms.VPSet(), #_rbxAndHPDCleaner # not needed for SiPM HCAL
    seedFinder = _localMaxSeeds_HCAL,
    initialClusteringStep = _topoClusterizer_HCAL,
    pfClusterBuilder = cms.PSet(), #_pfClusterizer_HCAL,
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

