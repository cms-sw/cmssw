from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from HIPixelTripletSeeds_cff import *
from HIPixel3PrimTracks_cfi import *

hiDetachedTripletStepClusters = cms.EDProducer("HITrackClusterRemover",
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
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
hiDetachedTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
hiDetachedTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiDetachedTripletStepClusters')
hiDetachedTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiDetachedTripletStepClusters')

# SEEDS
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

hiDetachedTripletStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
    precise = True,
    useMultipleScattering = False,
    useFakeVertices       = False,
    beamSpot = "offlineBeamSpot",
    useFixedError = True,
    nSigmaZ = 4.0,
    sigmaZVertex = 4.0,
    fixedError = 0.5,
    VertexCollection = "hiSelectedVertex",
    ptMin = 0.9,
    useFoundVertices = True,
    originRadius = 0.5
))
hiDetachedTripletStepTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "hiDetachedTripletStepSeedLayers",
    trackingRegions = "hiDetachedTripletStepTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
hiDetachedTripletStepTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "hiDetachedTripletStepTracksHitDoublets",
    extraHitRPhitolerance = 0.0,
    extraHitRZtolerance = 0.0,
    maxElement = 1000000,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    produceSeedingHitSets = True,
)
hiDetachedTripletStepPixelTracksFilter = hiFilter.clone(
    nSigmaTipMaxTolerance = 0,
    lipMax = 1.0,
    tipMax = 1.0,
    ptMin = 0.95,
)
hiDetachedTripletStepPixelTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel detached tracks with vertex constraint'),

    # Ordered Hits
    SeedingHitSets = cms.InputTag("hiDetachedTripletStepTracksHitTriplets"),
	
    # Fitter
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
	
    # Filter
    Filter = cms.InputTag("hiDetachedTripletStepPixelTracksFilter"),
	
    # Cleaner
    Cleaner = cms.string("trackCleaner")
)


import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
hiDetachedTripletStepSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
        InputCollection = 'hiDetachedTripletStepPixelTracks'
  )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiDetachedTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 6,
    minPt = cms.double(0.3),
    constantValueForLostHitsFractionFilter = cms.double(0.701)
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiDetachedTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
        ComponentName = cms.string('hiDetachedTripletStepChi2Est'),
            nSigma = cms.double(3.0),
            MaxChi2 = cms.double(9.0)
        )


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiDetachedTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiDetachedTripletStepTrajectoryFilter')),
    maxCand = 2,
    estimator = cms.string('hiDetachedTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(0),
    maxPtForLooperReconstruction = cms.double(0),
    alwaysUseInvalidHits = cms.bool(False)
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiDetachedTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiDetachedTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiDetachedTripletStepTrajectoryBuilder')),
    TrajectoryBuilder = cms.string('hiDetachedTripletStepTrajectoryBuilder'),
    clustersToSkip = cms.InputTag('hiDetachedTripletStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiDetachedTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiDetachedTripletStepTrackCandidates',
    AlgorithmName = cms.string('detachedTripletStep'),
    Fitter=cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiDetachedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiDetachedTripletStepTracks',
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('HIMVASelectorIter7'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiDetachedTripletStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiDetachedTripletStepTight',
    preFilterName = 'hiDetachedTripletStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.2)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiDetachedTripletStep',
    preFilterName = 'hiDetachedTripletStepTight',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.09)
    ),
    ) #end of vpset
    ) #end of clone

from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiDetachedTripletStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers=cms.VInputTag(cms.InputTag('hiDetachedTripletStepTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiDetachedTripletStepSelector","hiDetachedTripletStep")),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
    )


hiDetachedTripletStep = cms.Sequence(hiDetachedTripletStepClusters*
                                     hiDetachedTripletStepSeedLayers*
                                     hiDetachedTripletStepTrackingRegions*
                                     hiDetachedTripletStepTracksHitDoublets*
                                     hiDetachedTripletStepTracksHitTriplets*
                                     pixelFitterByHelixProjections*
                                     hiDetachedTripletStepPixelTracksFilter*
                                     hiDetachedTripletStepPixelTracks*
                                     hiDetachedTripletStepSeeds*
                                     hiDetachedTripletStepTrackCandidates*
                                     hiDetachedTripletStepTracks*
                                     hiDetachedTripletStepSelector*
                                     hiDetachedTripletStepQual)


