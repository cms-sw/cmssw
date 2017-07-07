from RecoTracker.IterativeTracking.LowPtQuadStep_cff import *
from HIPixelTripletSeeds_cff import *
from HIPixel3PrimTracks_cfi import *

hiLowPtQuadStepClusters = cms.EDProducer("HITrackClusterRemover",
     clusterLessSolution = cms.bool(True),
     trajectories = cms.InputTag("hiGlobalPrimTracks"),
     overrideTrkQuals = cms.InputTag('hiInitialStepSelector','hiInitialStep'),
     TrackQuality = cms.string('highPurity'),
     minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
     pixelClusters = cms.InputTag("siPixelClusters"),
     stripClusters = cms.InputTag("siStripClusters"),
     Common = cms.PSet(
         maxChi2 = cms.double(9.0),
     ),
     Strip = cms.PSet(
        #Yen-Jie's mod to preserve merged clusters
        maxSize = cms.uint32(2),
        maxChi2 = cms.double(9.0)
     )
)

# SEEDING LAYERS
# Using 4 layers layerlist
hiLowPtQuadStepSeedLayers = hiPixelLayerQuadruplets.clone()
hiLowPtQuadStepSeedLayers.BPix.skipClusters = cms.InputTag('hiLowPtQuadStepClusters')
hiLowPtQuadStepSeedLayers.FPix.skipClusters = cms.InputTag('hiLowPtQuadStepClusters')

# SEEDS
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelTriplets.pixelQuadrupletEDProducer_cfi import pixelQuadrupletEDProducer as _pixelQuadrupletEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

hiLowPtQuadStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
    precise = True,
    useMultipleScattering = False,
    useFakeVertices       = False,
    beamSpot = "offlineBeamSpot",
    useFixedError = True,
    nSigmaZ = 4.0,
    sigmaZVertex = 4.0,
    fixedError = 0.5,
    VertexCollection = "hiSelectedVertex",
    ptMin = 0.3,#0.2 for pp
    useFoundVertices = True,
    originRadius = 0.02 #0.02 for pp
))
hiLowPtQuadStepTracksHitDoubletsCA = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "hiLowPtQuadStepSeedLayers",
    trackingRegions = "hiLowPtQuadStepTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
    layerPairs = [0,1,2]
)

import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
hiLowPtQuadStepTracksHitQuadrupletsCA = _caHitQuadrupletEDProducer.clone(
    doublets = "hiLowPtQuadStepTracksHitDoubletsCA",
    extraHitRPhitolerance = 0.0,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 1000, value2 = 150,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0017,
    CAPhiCut = 0.3,
)


hiLowPtQuadStepPixelTracksFilter = hiFilter.clone(
    nSigmaTipMaxTolerance = 0,
    lipMax = 1.0,
    tipMax = 1.0,
    ptMin = 0.4, #seeding region is 0.3
)
hiLowPtQuadStepPixelTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel detached tracks with vertex constraint'),

    # Ordered Hits
    SeedingHitSets = cms.InputTag("hiLowPtQuadStepTracksHitQuadrupletsCA"),
	
    # Fitter
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
	
    # Filter
    Filter = cms.InputTag("hiLowPtQuadStepPixelTracksFilter"),
	
    # Cleaner
    Cleaner = cms.string("trackCleaner")
)


import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
hiLowPtQuadStepSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
        InputCollection = 'hiLowPtQuadStepPixelTracks'
  )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiLowPtQuadStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 1,
    minimumNumberOfHits = 3,#3 for pp
    minPt = cms.double(0.075),# 0.075 for pp
    #constantValueForLostHitsFractionFilter = cms.double(0.701)
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiLowPtQuadStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
        ComponentName = cms.string('hiLowPtQuadStepChi2Est'),
            nSigma = cms.double(3.0),
            MaxChi2 = cms.double(9.0)
        )


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiLowPtQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiLowPtQuadStepTrajectoryFilter')),
    maxCand = 4,#4 for pp
    estimator = cms.string('hiLowPtQuadStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),#2.0 for pp
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7),# 0.7 for pp
    alwaysUseInvalidHits = cms.bool(False)
    )

# MAKING OF TRACK CANDIDATES

# Trajectory cleaner in default

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiLowPtQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiLowPtQuadStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiLowPtQuadStepTrajectoryBuilder')),
    TrajectoryBuilder = cms.string('hiLowPtQuadStepTrajectoryBuilder'),
    clustersToSkip = cms.InputTag('hiLowPtQuadStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiLowPtQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiLowPtQuadStepTrackCandidates',
    AlgorithmName = cms.string('lowPtQuadStep'),
    Fitter=cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiLowPtQuadStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiLowPtQuadStepTracks',
    useAnyMVA = cms.bool(True), 
    GBRForestLabel = cms.string('HIMVASelectorIter8'),#FIXME MVA for new iteration
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiLowPtQuadStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiLowPtQuadStepTight',
    preFilterName = 'hiLowPtQuadStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.2)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiLowPtQuadStep',
    preFilterName = 'hiLowPtQuadStepTight',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.09)
    ),
    ) #end of vpset
    ) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiLowPtQuadStepSelector, useAnyMVA = cms.bool(False))
trackingPhase1.toModify(hiLowPtQuadStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiLowPtQuadStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiLowPtQuadStepTight',
    preFilterName = 'hiLowPtQuadStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.2)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiLowPtQuadStep',
    preFilterName = 'hiLowPtQuadStepTight',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.09)
    ),
    ) #end of vpset
)

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiLowPtQuadStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers=cms.VInputTag(cms.InputTag('hiLowPtQuadStepTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiLowPtQuadStepSelector","hiLowPtQuadStep")),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
    )


hiLowPtQuadStep = cms.Sequence(hiLowPtQuadStepClusters*
                                     hiLowPtQuadStepSeedLayers*
                                     hiLowPtQuadStepTrackingRegions*
                                     hiLowPtQuadStepTracksHitDoubletsCA* 
                                     hiLowPtQuadStepTracksHitQuadrupletsCA* 
				     pixelFitterByHelixProjections*
                                     hiLowPtQuadStepPixelTracksFilter*
                                     hiLowPtQuadStepPixelTracks*
                                     hiLowPtQuadStepSeeds*
                                     hiLowPtQuadStepTrackCandidates*
                                     hiLowPtQuadStepTracks*
                                     hiLowPtQuadStepSelector*
                                     hiLowPtQuadStepQual)


