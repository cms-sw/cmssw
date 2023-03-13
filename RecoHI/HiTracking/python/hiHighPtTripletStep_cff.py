from __future__ import absolute_import
from RecoTracker.IterativeTracking.HighPtTripletStep_cff import *
from .HIPixelTripletSeeds_cff import *
from .HIPixel3PrimTracks_cfi import *

hiHighPtTripletStepClusters = cms.EDProducer("HITrackClusterRemover",
     clusterLessSolution = cms.bool(True),
     trajectories = cms.InputTag("hiLowPtQuadStepTracks"),
     overrideTrkQuals = cms.InputTag("hiLowPtQuadStepSelector","hiLowPtQuadStep"),
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
# Using 3 layers layerlist
hiHighPtTripletStepSeedLayers = highPtTripletStepSeedLayers.clone(
    BPix = dict(skipClusters = 'hiHighPtTripletStepClusters'),
    FPix = dict(skipClusters = 'hiHighPtTripletStepClusters')
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

hiHighPtTripletStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
    precise = True,
    useMultipleScattering = False,
    useFakeVertices       = False,
    beamSpot = "offlineBeamSpot",
    useFixedError = True,
    nSigmaZ = 4.0,
    sigmaZVertex = 4.0,
    fixedError = 0.5,
    VertexCollection = "hiSelectedPixelVertex",
    ptMin = 0.8,#0.6 for pp
    useFoundVertices = True,
    originRadius = 0.02 #0.02 for pp
))
hiHighPtTripletStepTracksHitDoubletsCA = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "hiHighPtTripletStepSeedLayers",
    trackingRegions = "hiHighPtTripletStepTrackingRegions",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
    layerPairs = [0,1]
)

from RecoTracker.PixelSeeding.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
hiHighPtTripletStepTracksHitTripletsCA = _caHitTripletEDProducer.clone(
    doublets = "hiHighPtTripletStepTracksHitDoubletsCA",
    extraHitRPhitolerance = 0.0,
    SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 8,
        value1 = 100, value2 = 6,
    ),
    useBendingCorrection = True,
    CAThetaCut = 0.004,
    CAPhiCut = 0.07,
    CAHardPtCut = 0.3,
)

hiHighPtTripletStepPixelTracksFilter = hiFilter.clone(
    nSigmaTipMaxTolerance = 0,
    lipMax = 1.0,
    tipMax = 1.0,
    ptMin = 1.0, #seeding region is 0.6
)

import RecoTracker.PixelTrackFitting.pixelTracks_cfi as _mod

hiHighPtTripletStepPixelTracks = _mod.pixelTracks.clone(
    passLabel  = 'Pixel detached tracks with vertex constraint',
    # Ordered Hits
    SeedingHitSets = "hiHighPtTripletStepTracksHitTripletsCA",
    # Fitter
    Fitter = "pixelFitterByHelixProjections",
    # Filter
    Filter = "hiHighPtTripletStepPixelTracksFilter",
    # Cleaner
    Cleaner = "trackCleaner"
)

import RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi
hiHighPtTripletStepSeeds = RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
        InputCollection = 'hiHighPtTripletStepPixelTracks'
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiHighPtTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 1,
    minimumNumberOfHits = 3,#3 for pp
    minPt = 0.2,# 0.2 for pp
    #constantValueForLostHitsFractionFilter = 0.701
)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiHighPtTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'hiHighPtTripletStepChi2Est',
    nSigma  = 3.0,
    MaxChi2 = 9.0# 30 for pp
)


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiHighPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'hiHighPtTripletStepTrajectoryFilter'),
    maxCand = 3, # 3 for pp
    estimator = 'hiHighPtTripletStepChi2Est',
    maxDPhiForLooperReconstruction = 2.0, # 2.0 for pp
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (B=3.8T)
    maxPtForLooperReconstruction = 0.7, # 0.7 for pp
    alwaysUseInvalidHits = False
)

# MAKING OF TRACK CANDIDATES

# Trajectory cleaner in default

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiHighPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'hiHighPtTripletStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'hiHighPtTripletStepTrajectoryBuilder'),
    clustersToSkip = 'hiHighPtTripletStepClusters',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiHighPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiHighPtTripletStepTrackCandidates',
    AlgorithmName = 'highPtTripletStep',
    Fitter = 'FlexibleKFFittingSmoother'
)

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiHighPtTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiHighPtTripletStepTracks',
    useAnyMVA = True, 
    GBRForestLabel = 'HIMVASelectorIter9',#FIXME MVA for new iteration
    GBRForestVars  = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
    trackSelectors = cms.VPSet(
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
           name = 'hiHighPtTripletStepLoose',
           applyAdaptedPVCuts = False,
           useMVA = False,
       ), #end of pset
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
           name = 'hiHighPtTripletStepTight',
           preFilterName = 'hiHighPtTripletStepLoose',
           applyAdaptedPVCuts = False,
           useMVA = True,
           minMVA = -0.2
       ),
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
           name = 'hiHighPtTripletStep',
           preFilterName = 'hiHighPtTripletStepTight',
           applyAdaptedPVCuts = False,
           useMVA = True,
           minMVA = -0.09
       ),
    ) #end of vpset
) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiHighPtTripletStepSelector, useAnyMVA = False)
trackingPhase1.toModify(hiHighPtTripletStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
        name = 'hiHighPtTripletStepLoose',
        applyAdaptedPVCuts = False,
        useMVA = False,
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
        name = 'hiHighPtTripletStepTight',
        preFilterName = 'hiHighPtTripletStepLoose',
        applyAdaptedPVCuts = False,
        useMVA = False,
        minMVA = -0.2
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
        name = 'hiHighPtTripletStep',
        preFilterName = 'hiHighPtTripletStepTight',
        applyAdaptedPVCuts = False,
        useMVA = False,
        minMVA = -0.09
    ),
  ) #end of vpset
)

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiHighPtTripletStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiHighPtTripletStepTracks'],
    hasSelector = [1],
    selectedTrackQuals = ["hiHighPtTripletStepSelector:hiHighPtTripletStep"],
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
)


hiHighPtTripletStepTask = cms.Task(hiHighPtTripletStepClusters,
                                     hiHighPtTripletStepSeedLayers,
                                     hiHighPtTripletStepTrackingRegions,
                                     hiHighPtTripletStepTracksHitDoubletsCA, 
                                     hiHighPtTripletStepTracksHitTripletsCA, 
				     pixelFitterByHelixProjections,
                                     hiHighPtTripletStepPixelTracksFilter,
                                     hiHighPtTripletStepPixelTracks,
                                     hiHighPtTripletStepSeeds,
                                     hiHighPtTripletStepTrackCandidates,
                                     hiHighPtTripletStepTracks,
                                     hiHighPtTripletStepSelector,
                                     hiHighPtTripletStepQual)
hiHighPtTripletStep = cms.Sequence(hiHighPtTripletStepTask)

