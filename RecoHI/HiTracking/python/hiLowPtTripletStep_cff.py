import FWCore.ParameterSet.Config as cms


# NEW CLUSTERS (remove previously used clusters)
hiLowPtTripletStepClusters = cms.EDProducer("HITrackClusterRemover",
                                              clusterLessSolution= cms.bool(True),
                                              oldClusterRemovalInfo = cms.InputTag("hiDetachedTripletStepClusters"),
                                              trajectories = cms.InputTag("hiDetachedTripletStepTracks"),
                                              overrideTrkQuals = cms.InputTag("hiDetachedTripletStepSelector","hiDetachedTripletStep"),
                                              TrackQuality = cms.string('highPurity'),
                                              minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
                                              pixelClusters = cms.InputTag("siPixelClusters"),
                                              stripClusters = cms.InputTag("siStripClusters"),
                                              Common = cms.PSet(
        maxChi2 = cms.double(9.0)
        ),
                                              Strip = cms.PSet(
        #Yen-Jie's mod to preserve merged clusters
        maxSize = cms.uint32(2),
        maxChi2 = cms.double(9.0)
        )
                                              )


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
hiLowPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix = dict(skipClusters = cms.InputTag('hiLowPtTripletStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('hiLowPtTripletStepClusters'))
)

# SEEDS
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
from RecoTracker.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoTracker.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

hiLowPtTripletStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
    precise = True,
    useMultipleScattering = False,
    useFakeVertices       = False,
    beamSpot = "offlineBeamSpot",
    useFixedError = False,
    nSigmaZ = 4.0,
    sigmaZVertex = 4.0,
    fixedError = 0.2,
    VertexCollection = "hiSelectedPixelVertex",
    ptMin = 0.4,
    useFoundVertices = True,
    originRadius = 0.02
))
hiLowPtTripletStepTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "hiLowPtTripletStepSeedLayers",
    trackingRegions = "hiLowPtTripletStepTrackingRegions",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
)
import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
hiLowPtTripletStepTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "hiLowPtTripletStepTracksHitDoublets",
    #maxElement = 5000000,
    SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    produceSeedingHitSets = True,
)

from RecoTracker.PixelSeeding.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
hiLowPtTripletStepTracksHitDoubletsCA = hiLowPtTripletStepTracksHitDoublets.clone(
    layerPairs = [0,1]
)
hiLowPtTripletStepTracksHitTripletsCA = _caHitTripletEDProducer.clone(
    doublets = "hiLowPtTripletStepTracksHitDoubletsCA",
    extraHitRPhitolerance = hiLowPtTripletStepTracksHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = hiLowPtTripletStepTracksHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 70 , value2 = 8,
    ),
    useBendingCorrection = True,
    CAThetaCut = 0.002,
    CAPhiCut = 0.05,
)

hiLowPtTripletStepPixelTracksFilter = hiFilter.clone(
    nSigmaLipMaxTolerance = 4.0,
    nSigmaTipMaxTolerance = 4.0,
    lipMax = 0,
    ptMin = 0.4,
)

import RecoTracker.PixelTrackFitting.pixelTracks_cfi as _mod

hiLowPtTripletStepPixelTracks = _mod.pixelTracks.clone(
    passLabel  = 'Pixel primary tracks with vertex constraint',
    # Ordered Hits
    SeedingHitSets = "hiLowPtTripletStepTracksHitTriplets",
    # Fitter
    Fitter = "pixelFitterByHelixProjections",
    # Filter
    Filter = "hiLowPtTripletStepPixelTracksFilter",
    # Cleaner
    Cleaner = "trackCleaner"
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiLowPtTripletStepPixelTracks,
    SeedingHitSets = "hiLowPtTripletStepTracksHitTripletsCA"
)


import RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi
hiLowPtTripletStepSeeds = RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
        InputCollection = 'hiLowPtTripletStepPixelTracks'
)


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiLowPtTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 6,
    minPt = 0.4
)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiLowPtTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
        ComponentName = 'hiLowPtTripletStepChi2Est',
            nSigma  = 3.0,
            MaxChi2 = 9.0
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiLowPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'hiLowPtTripletStepTrajectoryFilter'),
    maxCand = 3,
    estimator = 'hiLowPtTripletStepChi2Est',
    maxDPhiForLooperReconstruction = 2.0,
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = 0.7,
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiLowPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'hiLowPtTripletStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'hiLowPtTripletStepTrajectoryBuilder'),
    clustersToSkip = 'hiLowPtTripletStepClusters',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiLowPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiLowPtTripletStepTrackCandidates',
    AlgorithmName = 'lowPtTripletStep',
    Fitter='FlexibleKFFittingSmoother'
)



# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiLowPtTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiLowPtTripletStepTracks',
    useAnyMVA = True,
    GBRForestLabel = 'HIMVASelectorIter5',
    GBRForestVars = ['chi2perdofperlayer', 'dxyperdxyerror', 'dzperdzerror', 'relpterr', 'nhits', 'nlayers', 'eta'],
    trackSelectors= cms.VPSet(
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
           name = 'hiLowPtTripletStepLoose',
           useMVA = False
       ), #end of pset
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
           name = 'hiLowPtTripletStepTight',
           preFilterName = 'hiLowPtTripletStepLoose',
           useMVA = True,
           minMVA = -0.58
       ),
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
           name = 'hiLowPtTripletStep',
           preFilterName = 'hiLowPtTripletStepTight',
           useMVA = True,
           minMVA = 0.35
       ),
    ) #end of vpset
) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiLowPtTripletStepSelector, useAnyMVA = False)
trackingPhase1.toModify(hiLowPtTripletStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
        name = 'hiLowPtTripletStepLoose',
        useMVA = False
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
        name = 'hiLowPtTripletStepTight',
        preFilterName = 'hiLowPtTripletStepLoose',
        useMVA = False,
        minMVA = -0.58
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
        name = 'hiLowPtTripletStep',
        preFilterName = 'hiLowPtTripletStepTight',
        useMVA = False,
        minMVA = 0.35
    ),
  ) #end of vpset
)


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiLowPtTripletStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiLowPtTripletStepTracks'],
    hasSelector = [1],
    selectedTrackQuals = ["hiLowPtTripletStepSelector:hiLowPtTripletStep"],
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
    #writeOnlyTrkQuals = True
)

# Final sequence

hiLowPtTripletStepTask = cms.Task(hiLowPtTripletStepClusters,
                                        hiLowPtTripletStepSeedLayers,
                                        hiLowPtTripletStepTrackingRegions,
                                        hiLowPtTripletStepTracksHitDoublets,
                                        hiLowPtTripletStepTracksHitTriplets,
                                        pixelFitterByHelixProjections,
                                        hiLowPtTripletStepPixelTracksFilter,
                                        hiLowPtTripletStepPixelTracks,hiLowPtTripletStepSeeds,
                                        hiLowPtTripletStepTrackCandidates,
                                        hiLowPtTripletStepTracks,
                                        hiLowPtTripletStepSelector,
                                        hiLowPtTripletStepQual
                                        )
hiLowPtTripletStep = cms.Sequence(hiLowPtTripletStepTask)
hiLowPtTripletStepTask_Phase1 = hiLowPtTripletStepTask.copy()
hiLowPtTripletStepTask_Phase1.replace(hiLowPtTripletStepTracksHitDoublets, hiLowPtTripletStepTracksHitDoubletsCA)
hiLowPtTripletStepTask_Phase1.replace(hiLowPtTripletStepTracksHitTriplets, hiLowPtTripletStepTracksHitTripletsCA)
trackingPhase1.toReplaceWith(hiLowPtTripletStepTask, hiLowPtTripletStepTask_Phase1)
