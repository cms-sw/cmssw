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
hiDetachedQuadStepSeedLayers = hiPixelLayerQuadruplets.clone()
hiDetachedQuadStepSeedLayers.BPix.skipClusters = cms.InputTag('hiDetachedQuadStepClusters')
hiDetachedQuadStepSeedLayers.FPix.skipClusters = cms.InputTag('hiDetachedQuadStepClusters')

# SEEDS
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

hiDetachedQuadStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
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

))
hiDetachedQuadStepTracksHitDoubletsCA = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "hiDetachedQuadStepSeedLayers",
    trackingRegions = "hiDetachedQuadStepTrackingRegions",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
    layerPairs = [0,1,2]
)

from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
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
hiDetachedQuadStepPixelTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel detached tracks with vertex constraint'),

    # Ordered Hits
    SeedingHitSets = cms.InputTag("hiDetachedQuadStepTracksHitQuadrupletsCA"),
	
    # Fitter
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
	
    # Filter
    Filter = cms.InputTag("hiDetachedQuadStepPixelTracksFilter"),
	
    # Cleaner
    Cleaner = cms.string("trackCleaner")
)


import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
hiDetachedQuadStepSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
        InputCollection = 'hiDetachedQuadStepPixelTracks'
  )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiDetachedQuadStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 1,
    minimumNumberOfHits = 3,#3 for pp
    minPt = cms.double(0.075),# 0.075 for pp
    #constantValueForLostHitsFractionFilter = cms.double(0.701)
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiDetachedQuadStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
        ComponentName = cms.string('hiDetachedQuadStepChi2Est'),
            nSigma = cms.double(3.0),
            MaxChi2 = cms.double(9.0)
        )


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiDetachedQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiDetachedQuadStepTrajectoryFilter')),
    maxCand = 4,#4 for pp
    estimator = cms.string('hiDetachedQuadStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),#2.0 for pp
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7),# 0.7 for pp
    alwaysUseInvalidHits = cms.bool(False)
    )

# MAKING OF TRACK CANDIDATES

# Trajectory cleaner in default

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiDetachedQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiDetachedQuadStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiDetachedQuadStepTrajectoryBuilder')),
    TrajectoryBuilder = cms.string('hiDetachedQuadStepTrajectoryBuilder'),
    clustersToSkip = cms.InputTag('hiDetachedQuadStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiDetachedQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiDetachedQuadStepTrackCandidates',
    AlgorithmName = cms.string('detachedQuadStep'),
    Fitter=cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiDetachedQuadStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiDetachedQuadStepTracks',
    useAnyMVA = cms.bool(True), 
    GBRForestLabel = cms.string('HIMVASelectorIter10'),#FIXME MVA for new iteration
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiDetachedQuadStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiDetachedQuadStepTight',
    preFilterName = 'hiDetachedQuadStepLoose',
    applyAdaptedPVCuts = cms.bool(True),
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.2)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiDetachedQuadStep',
    preFilterName = 'hiDetachedQuadStepTight',
    applyAdaptedPVCuts = cms.bool(True),
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.09)
    ),
    ) #end of vpset
    ) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiDetachedQuadStepSelector, useAnyMVA = cms.bool(False))
trackingPhase1.toModify(hiDetachedQuadStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiDetachedQuadStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiDetachedQuadStepTight',
    preFilterName = 'hiDetachedQuadStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.2)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiDetachedQuadStep',
    preFilterName = 'hiDetachedQuadStepTight',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.09)
    ),
    ) #end of vpset
)

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiDetachedQuadStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers=cms.VInputTag(cms.InputTag('hiDetachedQuadStepTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiDetachedQuadStepSelector","hiDetachedQuadStep")),
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

