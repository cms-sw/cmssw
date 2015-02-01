import FWCore.ParameterSet.Config as cms


# NEW CLUSTERS (remove previously used clusters)
hiSecondPixelTripletClusters = cms.EDProducer("HITrackClusterRemover",
                                              clusterLessSolution= cms.bool(True),
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
hiSecondPixelTripletSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
hiSecondPixelTripletSeedLayers.BPix.skipClusters = cms.InputTag('hiSecondPixelTripletClusters')
hiSecondPixelTripletSeedLayers.FPix.skipClusters = cms.InputTag('hiSecondPixelTripletClusters')

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cfi import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
hiSecondPixelTracks = cms.EDProducer("PixelTrackProducer",

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

hiSecondPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(5000000)
hiSecondPixelTracks.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('hiSecondPixelTripletSeedLayers')

hiSecondPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor



import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
hiSecondPixelTrackSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
        InputCollection = 'hiSecondPixelTracks'
  )



import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
hiSecondPixelTripletSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
	RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(False), #def value True
            nSigmaZ = cms.double(4.0),
            sigmaZVertex = cms.double(4.0), #def value 3
            fixedError = cms.double(0.2),
            VertexCollection = cms.InputTag("hiSelectedVertex"),
            ptMin = cms.double(0.4),
            useFoundVertices = cms.bool(True),
            originRadius = cms.double(0.02)
        )
    )
)

hiSecondPixelTripletSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiSecondPixelTripletSeedLayers'
hiSecondPixelTripletSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = 5000000
hiSecondPixelTripletSeeds.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000000
hiSecondPixelTripletSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 50000000
del hiSecondPixelTripletSeeds.ClusterCheckPSet.cut

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
hiSecondPixelTripletSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiSecondPixelTripletTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 6,
    minPt = 0.4
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
hiSecondPixelTripletChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
        ComponentName = cms.string('hiSecondPixelTripletChi2Est'),
            nSigma = cms.double(3.0),
            MaxChi2 = cms.double(9.0)
        )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiSecondPixelTripletTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiSecondPixelTripletTrajectoryFilter')),
    maxCand = 3,
    estimator = cms.string('hiSecondPixelTripletChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)  
    maxPtForLooperReconstruction = cms.double(0.7)
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiSecondPixelTripletTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiSecondPixelTrackSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiSecondPixelTripletTrajectoryBuilder')),
    clustersToSkip = cms.InputTag('hiSecondPixelTripletClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiSecondPixelTripletGlobalPrimTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiSecondPixelTripletTrackCandidates',
    AlgorithmName = cms.string('lowPtTripletStep'),
    Fitter=cms.string('FlexibleKFFittingSmoother')
    )



# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiSecondPixelTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiSecondPixelTripletGlobalPrimTracks',
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiSecondPixelTripletStepLoose',
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiSecondPixelTripletStepTight',
    preFilterName = 'hiSecondPixelTripletStepLoose',
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiSecondPixelTripletStep',
    preFilterName = 'hiSecondPixelTripletStepTight',
    # min_nhits = 14
    ),
    ) #end of vpset
    ) #end of clone


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiSecondQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('hiSecondPixelTripletGlobalPrimTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiSecondPixelTripletStepSelector","hiSecondPixelTripletStep")),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
    #writeOnlyTrkQuals = True
    )

# Final sequence

hiSecondPixelTripletStep = cms.Sequence(hiSecondPixelTripletClusters*
                                        hiSecondPixelTripletSeedLayers*
                                        hiSecondPixelTracks*hiSecondPixelTrackSeeds*
                                        hiSecondPixelTripletTrackCandidates*
                                        hiSecondPixelTripletGlobalPrimTracks*
                                        hiSecondPixelTripletStepSelector
                                        *hiSecondQual
                                        )
