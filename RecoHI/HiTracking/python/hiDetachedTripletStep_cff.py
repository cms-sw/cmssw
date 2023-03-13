from __future__ import absolute_import
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from .HIPixelTripletSeeds_cff import *
from .HIPixel3PrimTracks_cfi import *

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


from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiDetachedTripletStepClusters,
     trajectories     = "hiDetachedQuadStepTracks",
     overrideTrkQuals = "hiDetachedQuadStepSelector:hiDetachedQuadStep",
)


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
hiDetachedTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix = dict(skipClusters = cms.InputTag('hiDetachedTripletStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('hiDetachedTripletStepClusters'))
)

# SEEDS
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoTracker.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoTracker.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
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
    VertexCollection = "hiSelectedPixelVertex",
    ptMin = 0.9,
    useFoundVertices = True,
    originRadius = 0.5
))
hiDetachedTripletStepTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "hiDetachedTripletStepSeedLayers",
    trackingRegions = "hiDetachedTripletStepTrackingRegions",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
hiDetachedTripletStepTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "hiDetachedTripletStepTracksHitDoublets",
    extraHitRPhitolerance = 0.0,
    extraHitRZtolerance = 0.0,
    maxElement = 1000000,
    SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    produceSeedingHitSets = True,
)

from RecoTracker.PixelSeeding.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
hiDetachedTripletStepTracksHitDoubletsCA = hiDetachedTripletStepTracksHitDoublets.clone(
    layerPairs = [0,1]
)
hiDetachedTripletStepTracksHitTripletsCA = _caHitTripletEDProducer.clone(
    doublets = "hiDetachedTripletStepTracksHitDoubletsCA",
    extraHitRPhitolerance = hiDetachedTripletStepTracksHitTriplets.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 300 , value2 = 10,
    ),
    useBendingCorrection = True,
    CAThetaCut = 0.001,
    CAPhiCut = 0,
    CAHardPtCut = 0.2,
)

hiDetachedTripletStepPixelTracksFilter = hiFilter.clone(
    nSigmaTipMaxTolerance = 0,
    lipMax = 1.0,
    tipMax = 1.0,
    ptMin = 0.95,
)

import RecoTracker.PixelTrackFitting.pixelTracks_cfi as _mod

hiDetachedTripletStepPixelTracks = _mod.pixelTracks.clone(
    passLabel  = 'Pixel detached tracks with vertex constraint',
    # Ordered Hits
    SeedingHitSets = "hiDetachedTripletStepTracksHitTriplets",
    # Fitter
    Fitter = "pixelFitterByHelixProjections",
    # Filter
    Filter = "hiDetachedTripletStepPixelTracksFilter",
    # Cleaner
    Cleaner = "trackCleaner"
)
trackingPhase1.toModify(hiDetachedTripletStepPixelTracks,
    SeedingHitSets = "hiDetachedTripletStepTracksHitTripletsCA"
)


import RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi
hiDetachedTripletStepSeeds = RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
        InputCollection = 'hiDetachedTripletStepPixelTracks'
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiDetachedTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 6,
    minPt = 0.3,
    constantValueForLostHitsFractionFilter = 0.701
)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiDetachedTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
        ComponentName = 'hiDetachedTripletStepChi2Est',
            nSigma  = 3.0,
            MaxChi2 = 9.0
)


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiDetachedTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'hiDetachedTripletStepTrajectoryFilter'),
    maxCand = 2,
    estimator = 'hiDetachedTripletStepChi2Est',
    maxDPhiForLooperReconstruction = 0.,
    maxPtForLooperReconstruction   = 0.,
    alwaysUseInvalidHits = False
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiDetachedTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'hiDetachedTripletStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'hiDetachedTripletStepTrajectoryBuilder'),
    clustersToSkip = 'hiDetachedTripletStepClusters',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiDetachedTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiDetachedTripletStepTrackCandidates',
    AlgorithmName = 'detachedTripletStep',
    Fitter = 'FlexibleKFFittingSmoother'
)

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiDetachedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiDetachedTripletStepTracks',
    useAnyMVA = True,
    GBRForestLabel = 'HIMVASelectorIter7',
    GBRForestVars  = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
    trackSelectors = cms.VPSet(
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
           name = 'hiDetachedTripletStepLoose',
           applyAdaptedPVCuts = False,
           useMVA = False,
       ), #end of pset
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
           name = 'hiDetachedTripletStepTight',
           preFilterName = 'hiDetachedTripletStepLoose',
           applyAdaptedPVCuts = False,
           useMVA = True,
           minMVA = -0.2
       ),
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
           name = 'hiDetachedTripletStep',
           preFilterName = 'hiDetachedTripletStepTight',
           applyAdaptedPVCuts = False,
           useMVA = True,
           minMVA = -0.09
       ),
    ) #end of vpset
) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiDetachedTripletStepSelector, useAnyMVA = False)
trackingPhase1.toModify(hiDetachedTripletStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
        name = 'hiDetachedTripletStepLoose',
        applyAdaptedPVCuts = False,
        useMVA = False,
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
        name = 'hiDetachedTripletStepTight',
        preFilterName = 'hiDetachedTripletStepLoose',
        applyAdaptedPVCuts = False,
        useMVA = False,
        minMVA = -0.2
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
        name = 'hiDetachedTripletStep',
        preFilterName = 'hiDetachedTripletStepTight',
        applyAdaptedPVCuts = False,
        useMVA = False,
        minMVA = -0.09
    ),
  ) #end of vpset
)

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiDetachedTripletStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiDetachedTripletStepTracks'],
    hasSelector = [1],
    selectedTrackQuals = ["hiDetachedTripletStepSelector:hiDetachedTripletStep"],
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
)


hiDetachedTripletStepTask = cms.Task(hiDetachedTripletStepClusters,
                                     hiDetachedTripletStepSeedLayers,
                                     hiDetachedTripletStepTrackingRegions,
                                     hiDetachedTripletStepTracksHitDoublets,  
                                     hiDetachedTripletStepTracksHitTriplets, 
                                     pixelFitterByHelixProjections,
                                     hiDetachedTripletStepPixelTracksFilter,
                                     hiDetachedTripletStepPixelTracks,
                                     hiDetachedTripletStepSeeds,
                                     hiDetachedTripletStepTrackCandidates,
                                     hiDetachedTripletStepTracks,
                                     hiDetachedTripletStepSelector,
                                     hiDetachedTripletStepQual)
hiDetachedTripletStep = cms.Sequence(hiDetachedTripletStepTask)
hiDetachedTripletStepTask_Phase1 = hiDetachedTripletStepTask.copy()
hiDetachedTripletStepTask_Phase1.replace(hiDetachedTripletStepTracksHitDoublets, hiDetachedTripletStepTracksHitDoubletsCA)
hiDetachedTripletStepTask_Phase1.replace(hiDetachedTripletStepTracksHitTriplets, hiDetachedTripletStepTracksHitTripletsCA)
trackingPhase1.toReplaceWith(hiDetachedTripletStepTask, hiDetachedTripletStepTask_Phase1)
