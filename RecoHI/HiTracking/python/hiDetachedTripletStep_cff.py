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
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cfi import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
hiDetachedTripletStepPixelTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel detached tracks with vertex constraint'),

    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(True),
            nSigmaZ = cms.double(4.0),
            sigmaZVertex = cms.double(4.0),
            fixedError = cms.double(0.5),
            VertexCollection = cms.InputTag("hiSelectedVertex"),
            ptMin = cms.double(0.9),
            useFoundVertices = cms.bool(True),
            originRadius = cms.double(0.5)
        )
    ),
     
    # Ordered Hits
    OrderedHitsFactoryPSet = cms.PSet( 
          ComponentName = cms.string( "StandardHitTripletGenerator" ),
	  SeedingLayers = cms.InputTag( "PixelLayerTriplets" ),
          GeneratorPSet = cms.PSet( 
		PixelTripletHLTGenerator
          )
    ),
	
    # Fitter
    FitterPSet = cms.PSet( 
	  ComponentName = cms.string('PixelFitterByHelixProjections'),
	  TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    ),
	
    # Filter
    useFilterWithES = cms.bool( True ),
    FilterPSet = cms.PSet( 
        nSigmaLipMaxTolerance = cms.double(0),
        chi2 = cms.double(1000.0),
        ComponentName = cms.string('HIPixelTrackFilter'),
        nSigmaTipMaxTolerance = cms.double(0),
        clusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        VertexCollection = cms.InputTag("hiSelectedVertex"),
        useClusterShape = cms.bool(False),
        lipMax = cms.double(1.0),
        tipMax = cms.double(1.0),
        ptMin = cms.double(0.95)
    ),
	
    # Cleaner
    CleanerPSet = cms.PSet(  
          ComponentName = cms.string( "TrackCleaner" )
    )
)



hiDetachedTripletStepPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.extraHitRPhitolerance = cms.double(0.0)
hiDetachedTripletStepPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.extraHitRZtolerance = cms.double(0.0)
hiDetachedTripletStepPixelTracks.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('hiDetachedTripletStepSeedLayers')

hiDetachedTripletStepPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
hiDetachedTripletStepPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor

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

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
hiDetachedTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
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
                                     hiDetachedTripletStepPixelTracks*
                                     hiDetachedTripletStepSeeds*
                                     hiDetachedTripletStepTrackCandidates*
                                     hiDetachedTripletStepTracks*
                                     hiDetachedTripletStepSelector*
                                     hiDetachedTripletStepQual)


