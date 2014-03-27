import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
hiMixedTripletClusters = cms.EDProducer("TrackClusterRemover",
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
hiMixedTripletSeedLayersA = cms.EDProducer("SeedingLayersEDProducer",
                                             layerList = cms.vstring('FPix1_pos+FPix2_pos+TEC1_pos', 'FPix1_neg+FPix2_neg+TEC1_neg'),
                                                                     #'FPix2_pos+TEC2_pos+TEC3_pos', 'FPix2_neg+TEC2_neg+TEC3_neg'),
                                   BPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('hiMixedTripletClusters')
    ),
                                   FPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('hiMixedTripletClusters')
    ),
                                   TEC = cms.PSet(
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    minRing = cms.int32(1),
    maxRing = cms.int32(1),
    skipClusters = cms.InputTag('hiMixedTripletClusters')
    )
                                   )

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
hiMixedTripletSeedsA = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
hiMixedTripletSeedsA.OrderedHitsFactoryPSet.SeedingLayers = 'hiMixedTripletSeedLayersA'
hiMixedTripletSeedsA.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
hiMixedTripletSeedsA.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
hiMixedTripletSeedsA.RegionFactoryPSet.RegionPSet.ptMin = 4.0
hiMixedTripletSeedsA.RegionFactoryPSet.RegionPSet.originRadius = 0.005
#hiMixedTripletSeedsA.RegionFactoryPSet.RegionPSet.nSigmaZ = 4.0
hiMixedTripletSeedsA.RegionFactoryPSet.RegionPSet.originHalfLength = 10.0

hiMixedTripletSeedsA.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = 5000000
hiMixedTripletSeedsA.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000000
hiMixedTripletSeedsA.ClusterCheckPSet.MaxNumberOfCosmicClusters = 50000000


# SEEDING LAYERS
hiMixedTripletSeedLayersB = cms.EDProducer("SeedingLayersEDProducer",
                                   layerList = cms.vstring(
    #'BPix1+BPix2+TIB1',
    #'BPix1+BPix2+TIB2',    
    #'BPix1+BPix3+TIB1',
    #'BPix1+BPix3+TIB2',    
    'BPix2+BPix3+TIB1',
    'BPix2+BPix3+TIB2'),
                                   BPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('hiMixedTripletClusters')
    ),
                                   TIB = cms.PSet(
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    TTRHBuilder = cms.string('WithTrackAngle'),
    skipClusters = cms.InputTag('hiMixedTripletClusters')
    )
                                   )

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
hiMixedTripletSeedsB = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
hiMixedTripletSeedsB.OrderedHitsFactoryPSet.SeedingLayers = 'hiMixedTripletSeedLayersB'
hiMixedTripletSeedsB.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
hiMixedTripletSeedsB.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
hiMixedTripletSeedsB.RegionFactoryPSet.RegionPSet.ptMin = 4.0
hiMixedTripletSeedsB.RegionFactoryPSet.RegionPSet.originRadius = 0.005
#hiMixedTripletSeedsB.RegionFactoryPSet.RegionPSet.nSigmaZ = 4.0
hiMixedTripletSeedsB.RegionFactoryPSet.RegionPSet.originHalfLength = 10.0

hiMixedTripletSeedsB.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = 5000000
hiMixedTripletSeedsB.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000000
hiMixedTripletSeedsB.ClusterCheckPSet.MaxNumberOfCosmicClusters = 50000000

import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
hiMixedTripletSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
hiMixedTripletSeeds.seedCollections = cms.VInputTag(
    cms.InputTag('hiMixedTripletSeedsA'),
    cms.InputTag('hiMixedTripletSeedsB'),
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
hiMixedTripletTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'hiMixedTripletTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 6,
    minPt = 1.0
    )
    )

# Propagator taking into account momentum uncertainty in multiple scattering calculation.
import TrackingTools.MaterialEffects.MaterialPropagator_cfi
hiMixedTripletPropagator = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone(
    ComponentName = 'hiMixedTripletPropagator',
    ptMin = 1.0
    )
import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
hiMixedTripletPropagatorOpposite = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
    ComponentName = 'hiMixedTripletPropagatorOpposite',
    ptMin = 1.0
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
hiMixedTripletChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('hiMixedTripletChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(16.0)
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
hiMixedTripletTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'hiMixedTripletTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'hiMixedTripletTrajectoryFilter',
    propagatorAlong = cms.string('hiMixedTripletPropagator'),
    propagatorOpposite = cms.string('hiMixedTripletPropagatorOpposite'),
    clustersToSkip = cms.InputTag('hiMixedTripletClusters'),
    maxCand = 2,
    estimator = cms.string('hiMixedTripletChi2Est')
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiMixedTripletTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiMixedTripletSeeds'),
    TrajectoryBuilder = 'hiMixedTripletTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiMixedTripletGlobalPrimTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter4'),
    src = 'hiMixedTripletTrackCandidates'
    )

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiMixedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiMixedTripletGlobalPrimTracks',
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiMixedTripletStepLoose',
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiMixedTripletStepTight',
    preFilterName = 'hiMixedTripletStepLoose',
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiMixedTripletStep',
    preFilterName = 'hiMixedTripletStepTight',
    min_nhits = 14
    ),
    ) #end of vpset
    ) #end of clone



# Final sequence

hiMixedTripletStep = cms.Sequence(
                          hiMixedTripletClusters*
                          hiMixedTripletSeedLayersA*
                          hiMixedTripletSeedsA*
                          hiMixedTripletSeedLayersB*
                          hiMixedTripletSeedsB*
                          hiMixedTripletSeeds*
                          hiMixedTripletTrackCandidates*
                          hiMixedTripletGlobalPrimTracks*
                          hiMixedTripletStepSelector)

