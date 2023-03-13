from __future__ import absolute_import
from RecoTracker.IterativeTracking.DetachedQuadStep_cff import *
from .HIPixelTripletSeeds_cff import *
from .HIPixel3PrimTracks_cfi import *

hiDetachedQuadStepClusters = cms.EDProducer("HITrackClusterRemover",
     clusterLessSolution = cms.bool(True),
     trajectories = cms.InputTag("hiHighPtTripletStepTracks"),
     overrideTrkQuals = cms.InputTag("hiHighPtTripletStepSelector","hiHighPtTripletStep"),
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
hiDetachedQuadStepSeedLayers = hiPixelLayerQuadruplets.clone(
    BPix = dict(skipClusters = cms.InputTag('hiDetachedQuadStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('hiDetachedQuadStepClusters'))
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

hiDetachedQuadStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet = dict(
    precise = True,
    useMultipleScattering = False,
    useFakeVertices       = False,
    beamSpot = "offlineBeamSpot",
    useFixedError = True,
    nSigmaZ = 4.0,
    sigmaZVertex = 4.0,
    fixedError = 0.5,
    VertexCollection = "hiSelectedPixelVertex",
    ptMin = 0.9,# 0.3 for pp
    useFoundVertices = True,
    #originHalfLength = 15.0, # 15 for pp, useTrackingRegionWithVertices, does not have this parameter. Only with BeamSpot
    originRadius = 1.5 # 1.5 for pp
    )
)
hiDetachedQuadStepTracksHitDoubletsCA = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "hiDetachedQuadStepSeedLayers",
    trackingRegions = "hiDetachedQuadStepTrackingRegions",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
    layerPairs = [0,1,2]
)

from RecoTracker.PixelSeeding.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
hiDetachedQuadStepTracksHitQuadrupletsCA = _caHitQuadrupletEDProducer.clone(
    doublets = "hiDetachedQuadStepTracksHitDoubletsCA",
    extraHitRPhitolerance = 0.0,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 500, value2 = 100,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0011,
    CAPhiCut = 0,
)

hiDetachedQuadStepPixelTracksFilter = hiFilter.clone(
    nSigmaTipMaxTolerance = 0,
    lipMax = 1.0,
    tipMax = 1.0,
    ptMin = 0.95, #seeding region is 0.3
)

import RecoTracker.PixelTrackFitting.pixelTracks_cfi as _mod

hiDetachedQuadStepPixelTracks = _mod.pixelTracks.clone(
    passLabel  = 'Pixel detached tracks with vertex constraint',
    # Ordered Hits
    SeedingHitSets = "hiDetachedQuadStepTracksHitQuadrupletsCA",
    # Fitter
    Fitter = "pixelFitterByHelixProjections",
    # Filter
    Filter = "hiDetachedQuadStepPixelTracksFilter",
    # Cleaner
    Cleaner = "trackCleaner"
)


import RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi
hiDetachedQuadStepSeeds = RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
    InputCollection = 'hiDetachedQuadStepPixelTracks'
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiDetachedQuadStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 1,
    minimumNumberOfHits = 3,#3 for pp
    minPt = 0.075,# 0.075 for pp
    #constantValueForLostHitsFractionFilter = 0.701
)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiDetachedQuadStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'hiDetachedQuadStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 9.0
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiDetachedQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'hiDetachedQuadStepTrajectoryFilter'),
    maxCand = 4, # 4 for pp
    estimator = 'hiDetachedQuadStepChi2Est',
    maxDPhiForLooperReconstruction = 2.0, # 2.0 for pp
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (B=3.8T)
    maxPtForLooperReconstruction = 0.7, # 0.7 for pp
    alwaysUseInvalidHits = False
)

# MAKING OF TRACK CANDIDATES

# Trajectory cleaner in default

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiDetachedQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'hiDetachedQuadStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'hiDetachedQuadStepTrajectoryBuilder'),
    clustersToSkip = 'hiDetachedQuadStepClusters',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiDetachedQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiDetachedQuadStepTrackCandidates',
    AlgorithmName = 'detachedQuadStep',
    Fitter='FlexibleKFFittingSmoother'
)

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiDetachedQuadStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiDetachedQuadStepTracks',
    useAnyMVA = True, 
    GBRForestLabel = 'HIMVASelectorIter10',#FIXME MVA for new iteration
    GBRForestVars = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
    trackSelectors= cms.VPSet(
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiDetachedQuadStepLoose',
            applyAdaptedPVCuts = False,
            useMVA = False,
        ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiDetachedQuadStepTight',
            preFilterName = 'hiDetachedQuadStepLoose',
            applyAdaptedPVCuts = True,
            useMVA = True,
            minMVA = -0.2
        ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiDetachedQuadStep',
            preFilterName = 'hiDetachedQuadStepTight',
            applyAdaptedPVCuts = True,
            useMVA = True,
            minMVA = -0.09
        ),
    ) #end of vpset
) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiDetachedQuadStepSelector, 
    useAnyMVA = False, 
    trackSelectors= cms.VPSet(
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiDetachedQuadStepLoose',
            applyAdaptedPVCuts = False,
            useMVA = False,
        ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiDetachedQuadStepTight',
            preFilterName = 'hiDetachedQuadStepLoose',
            applyAdaptedPVCuts = False,
            useMVA = False,
            minMVA = -0.2
        ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiDetachedQuadStep',
            preFilterName = 'hiDetachedQuadStepTight',
            applyAdaptedPVCuts = False,
            useMVA = False,
            minMVA = -0.09
        ),
    ) #end of vpset
)

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiDetachedQuadStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiDetachedQuadStepTracks'],
    hasSelector = [1],
    selectedTrackQuals = ["hiDetachedQuadStepSelector:hiDetachedQuadStep"],
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
)


hiDetachedQuadStepTask = cms.Task(hiDetachedQuadStepClusters,
                                     hiDetachedQuadStepSeedLayers,
                                     hiDetachedQuadStepTrackingRegions,
                                     hiDetachedQuadStepTracksHitDoubletsCA, 
                                     hiDetachedQuadStepTracksHitQuadrupletsCA, 
				     pixelFitterByHelixProjections,
                                     hiDetachedQuadStepPixelTracksFilter,
                                     hiDetachedQuadStepPixelTracks,
                                     hiDetachedQuadStepSeeds,
                                     hiDetachedQuadStepTrackCandidates,
                                     hiDetachedQuadStepTracks,
                                     hiDetachedQuadStepSelector,
                                     hiDetachedQuadStepQual)
hiDetachedQuadStep = cms.Sequence(hiDetachedQuadStepTask)

