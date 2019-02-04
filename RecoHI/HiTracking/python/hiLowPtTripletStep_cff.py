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
hiLowPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
hiLowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiLowPtTripletStepClusters')
hiLowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiLowPtTripletStepClusters')

# SEEDS
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
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
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
hiLowPtTripletStepTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "hiLowPtTripletStepTracksHitDoublets",
    #maxElement = 5000000,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    produceSeedingHitSets = True,
)

from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
hiLowPtTripletStepTracksHitDoubletsCA = hiLowPtTripletStepTracksHitDoublets.clone()
hiLowPtTripletStepTracksHitDoubletsCA.layerPairs = [0,1]

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
hiLowPtTripletStepPixelTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel primary tracks with vertex constraint'),

    # Ordered Hits
    SeedingHitSets = cms.InputTag("hiLowPtTripletStepTracksHitTriplets"),
	
    # Fitter
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
	
    # Filter
    Filter = cms.InputTag("hiLowPtTripletStepPixelTracksFilter"),
	
    # Cleaner
    Cleaner = cms.string("trackCleaner")
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiLowPtTripletStepPixelTracks,
    SeedingHitSets = cms.InputTag("hiLowPtTripletStepTracksHitTripletsCA")
)


import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
hiLowPtTripletStepSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
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
        ComponentName = cms.string('hiLowPtTripletStepChi2Est'),
            nSigma = cms.double(3.0),
            MaxChi2 = cms.double(9.0)
        )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiLowPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiLowPtTripletStepTrajectoryFilter')),
    maxCand = 3,
    estimator = cms.string('hiLowPtTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)  
    maxPtForLooperReconstruction = cms.double(0.7)
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiLowPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiLowPtTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiLowPtTripletStepTrajectoryBuilder')),
    clustersToSkip = cms.InputTag('hiLowPtTripletStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiLowPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiLowPtTripletStepTrackCandidates',
    AlgorithmName = cms.string('lowPtTripletStep'),
    Fitter=cms.string('FlexibleKFFittingSmoother')
    )



# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiLowPtTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiLowPtTripletStepTracks',
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('HIMVASelectorIter5'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'dxyperdxyerror', 'dzperdzerror', 'relpterr', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiLowPtTripletStepLoose',
    useMVA = cms.bool(False)
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiLowPtTripletStepTight',
    preFilterName = 'hiLowPtTripletStepLoose',
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.58)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiLowPtTripletStep',
    preFilterName = 'hiLowPtTripletStepTight',
    useMVA = cms.bool(True),
    minMVA = cms.double(0.35)
    ),
    ) #end of vpset
    ) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiLowPtTripletStepSelector, useAnyMVA = cms.bool(False))
trackingPhase1.toModify(hiLowPtTripletStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiLowPtTripletStepLoose',
    useMVA = cms.bool(False)
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiLowPtTripletStepTight',
    preFilterName = 'hiLowPtTripletStepLoose',
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.58)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiLowPtTripletStep',
    preFilterName = 'hiLowPtTripletStepTight',
    useMVA = cms.bool(False),
    minMVA = cms.double(0.35)
    ),
    ) #end of vpset
)


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiLowPtTripletStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('hiLowPtTripletStepTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiLowPtTripletStepSelector","hiLowPtTripletStep")),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
    #writeOnlyTrkQuals = True
    )

# Final sequence

hiLowPtTripletStep = cms.Sequence(hiLowPtTripletStepClusters*
                                        hiLowPtTripletStepSeedLayers*
                                        hiLowPtTripletStepTrackingRegions*
                                        hiLowPtTripletStepTracksHitDoublets*
                                        hiLowPtTripletStepTracksHitTriplets*
                                        pixelFitterByHelixProjections*
                                        hiLowPtTripletStepPixelTracksFilter*
                                        hiLowPtTripletStepPixelTracks*hiLowPtTripletStepSeeds*
                                        hiLowPtTripletStepTrackCandidates*
                                        hiLowPtTripletStepTracks*
                                        hiLowPtTripletStepSelector*
                                        hiLowPtTripletStepQual
                                        )
hiLowPtTripletStep_Phase1 = hiLowPtTripletStep.copy()
hiLowPtTripletStep_Phase1.replace(hiLowPtTripletStepTracksHitDoublets, hiLowPtTripletStepTracksHitDoubletsCA)
hiLowPtTripletStep_Phase1.replace(hiLowPtTripletStepTracksHitTriplets, hiLowPtTripletStepTracksHitTripletsCA)
trackingPhase1.toReplaceWith(hiLowPtTripletStep, hiLowPtTripletStep_Phase1)
