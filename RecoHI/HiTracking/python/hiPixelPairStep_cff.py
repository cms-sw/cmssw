import FWCore.ParameterSet.Config as cms



# NEW CLUSTERS (remove previously used clusters)
hiPixelPairClusters = cms.EDProducer("TrackClusterRemover",
                                     clusterLessSolution= cms.bool(True),
                                     oldClusterRemovalInfo = cms.InputTag("hiSecondPixelTripletClusters"),
                                     trajectories = cms.InputTag("hiSecondPixelTripletGlobalPrimTracks"),
                                     overrideTrkQuals = cms.InputTag('hiSecondPixelTripletStepSelector','hiSecondPixelTripletStep'),
                                     TrackQuality = cms.string('highPurity'),
                                     pixelClusters = cms.InputTag("siPixelClusters"),
                                     stripClusters = cms.InputTag("siStripClusters"),
                                     Common = cms.PSet(
    maxChi2 = cms.double(9.0),
    ),
                                     Strip = cms.PSet(
    maxChi2 = cms.double(9.0),
    #Yen-Jie's mod to preserve merged clusters
    maxSize = cms.uint32(2)
    )
                                     )


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi
hiPixelPairSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi.PixelLayerPairs.clone(
            layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
                                    'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
                                    'BPix2+FPix1_pos', 'BPix2+FPix1_neg',
                                    'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg'),
            )

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff
hiPixelPairSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.VertexCollection=cms.InputTag("hiSelectedVertex")
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.ptMin = 4.0
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.originRadius = 0.005
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.nSigmaZ = 4.0
# sigmaZVertex is only used when usedFixedError is True -Matt
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.sigmaZVertex = 4.0
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.useFixedError = cms.bool(False)
hiPixelPairSeeds.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('hiPixelPairSeedLayers')
hiPixelPairSeeds.OrderedHitsFactoryPSet.maxElement = 5000000
hiPixelPairSeeds.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000000
hiPixelPairSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 50000000

hiPixelPairSeeds.SeedComparitorPSet = cms.PSet(
    ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
    FilterAtHelixStage = cms.bool(True),
    FilterPixelHits = cms.bool(True),
    FilterStripHits = cms.bool(False),
    ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
    ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache")
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiPixelPairTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 0,
    minimumNumberOfHits = 6,
    minPt = 1.0
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
hiPixelPairChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
        ComponentName = cms.string('hiPixelPairChi2Est'),
            nSigma = cms.double(3.0),
            MaxChi2 = cms.double(9.0)
        )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiPixelPairTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
        MeasurementTrackerName = '',
        trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiPixelPairTrajectoryFilter')),
        clustersToSkip = cms.InputTag('hiPixelPairClusters'),
        maxCand = 3,
        #estimator = cms.string('hiPixelPairChi2Est')
        )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiPixelPairTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiPixelPairSeeds'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiPixelPairTrajectoryBuilder'))
    )


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiPixelPairGlobalPrimTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiPixelPairTrackCandidates',
    AlgorithmName = cms.string('iter2')
    )



# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiPixelPairStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiPixelPairGlobalPrimTracks',
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiPixelPairStepLoose',
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiPixelPairStepTight',
    preFilterName = 'hiPixelPairStepLoose',
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiPixelPairStep',
    preFilterName = 'hiPixelPairStepTight',
    min_nhits = 14
    ),
    ) #end of vpset
    ) #end of clone



# Final sequence

hiPixelPairStep = cms.Sequence(hiPixelPairClusters*
                               hiPixelPairSeedLayers*
                               hiPixelPairSeeds*
                               hiPixelPairTrackCandidates*
                               hiPixelPairGlobalPrimTracks*
                               hiPixelPairStepSelector)
