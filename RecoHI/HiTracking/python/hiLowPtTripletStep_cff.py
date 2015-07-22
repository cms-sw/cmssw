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
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cfi import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
hiLowPtTripletStepPixelTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel primary tracks with vertex constraint'),

    # Region
    RegionFactoryPSet = cms.PSet(
	  ComponentName = cms.string("GlobalTrackingRegionWithVerticesProducer"),
	  RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(False),
            nSigmaZ = cms.double(4.0),
            sigmaZVertex = cms.double(4.0),
            fixedError = cms.double(0.2),
            VertexCollection = cms.InputTag("hiSelectedVertex"),
            ptMin = cms.double(0.4),
            useFoundVertices = cms.bool(True),
            originRadius = cms.double(0.02)
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
        nSigmaLipMaxTolerance = cms.double(4.0),
        chi2 = cms.double(1000.0),
        ComponentName = cms.string('HIPixelTrackFilter'),
        nSigmaTipMaxTolerance = cms.double(4.0),
        clusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        VertexCollection = cms.InputTag("hiSelectedVertex"),
        useClusterShape = cms.bool(False),
        lipMax = cms.double(0),
        tipMax = cms.double(0),
        ptMin = cms.double(0.4)
    ),
	
    # Cleaner
    CleanerPSet = cms.PSet(  
          ComponentName = cms.string( "TrackCleaner" )
    )
)

hiLowPtTripletStepPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(5000000)
hiLowPtTripletStepPixelTracks.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('hiLowPtTripletStepSeedLayers')

hiLowPtTripletStepPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor



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

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
hiLowPtTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
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
                                        hiLowPtTripletStepPixelTracks*hiLowPtTripletStepSeeds*
                                        hiLowPtTripletStepTrackCandidates*
                                        hiLowPtTripletStepTracks*
                                        hiLowPtTripletStepSelector*
                                        hiLowPtTripletStepQual
                                        )
