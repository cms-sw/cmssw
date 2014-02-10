import FWCore.ParameterSet.Config as cms

###############################################################
# Large impact parameter Tracking using mixed-triplet seeding #
###############################################################

mixedTripletStepClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution = cms.bool(True),
    oldClusterRemovalInfo = cms.InputTag("detachedTripletStepClusters"),
    trajectories = cms.InputTag("detachedTripletStepTracks"),
    overrideTrkQuals = cms.InputTag('detachedTripletStep'),
    TrackQuality = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    )
)

# SEEDING LAYERS
mixedTripletStepSeedLayersA = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg', 
        'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg', 
        'FPix1_pos+FPix2_pos+TEC1_pos', 'FPix1_neg+FPix2_neg+TEC1_neg',
        'FPix1_pos+FPix2_pos+TEC2_pos', 'FPix1_neg+FPix2_neg+TEC2_neg',
        'FPix2_pos+TEC2_pos+TEC3_pos', 'FPix2_neg+TEC2_neg+TEC3_neg'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    )
)

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
mixedTripletStepSeedsA = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
mixedTripletStepSeedsA.OrderedHitsFactoryPSet.SeedingLayers = 'mixedTripletStepSeedLayersA'
mixedTripletStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
mixedTripletStepSeedsA.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.ptMin = 0.3
mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originHalfLength = 20.0
mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originRadius = 1.5

# SEEDING LAYERS
mixedTripletStepSeedLayersB = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix2+BPix3+TIB1', 'BPix2+BPix3+TIB2'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    )
)

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
mixedTripletStepSeedsB = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
mixedTripletStepSeedsB.OrderedHitsFactoryPSet.SeedingLayers = 'mixedTripletStepSeedLayersB'
mixedTripletStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
mixedTripletStepSeedsB.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.ptMin = 0.4
mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.originRadius = 1.5

mixedTripletStepSeedsB.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(True),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
    )

import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
mixedTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
mixedTripletStepSeeds.seedCollections = cms.VInputTag(
        cms.InputTag('mixedTripletStepSeedsA'),
        cms.InputTag('mixedTripletStepSeedsB'),
        )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
mixedTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'mixedTripletStepTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 3,
    minPt = 0.05
    )
    )

# Propagator taking into account momentum uncertainty in multiple scattering calculation.
import TrackingTools.MaterialEffects.MaterialPropagator_cfi
mixedTripletStepPropagator = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone(
    ComponentName = 'mixedTripletStepPropagator',
    ptMin = 0.05
    )
import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
mixedTripletStepPropagatorOpposite = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
    ComponentName = 'mixedTripletStepPropagatorOpposite',
    ptMin = 0.05
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
mixedTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('mixedTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(25.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
mixedTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'mixedTripletStepTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'mixedTripletStepTrajectoryFilter',
    propagatorAlong = cms.string('mixedTripletStepPropagator'),
    propagatorOpposite = cms.string('mixedTripletStepPropagatorOpposite'),
    clustersToSkip = cms.InputTag('mixedTripletStepClusters'),
    maxCand = 3,
    estimator = cms.string('mixedTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
mixedTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('mixedTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    #onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = 'mixedTripletStepTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
mixedTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('mixedTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.11),
            allowSharedFirstHit = cms.bool(True)
            )
mixedTripletStepTrackCandidates.TrajectoryCleaner = 'mixedTripletStepTrajectoryCleanerBySharedHits'


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
mixedTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter4'),
    src = 'mixedTripletStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
)

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
mixedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='mixedTripletStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'mixedTripletStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.2, 3.0 ),
            dz_par1 = ( 1.2, 3.0 ),
            d0_par2 = ( 1.3, 3.0 ),
            dz_par2 = ( 1.3, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'mixedTripletStepTrkLoose',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.0, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'mixedTripletStepVtxTight',
            preFilterName = 'mixedTripletStepVtxLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.0, 3.0 ),
            dz_par1 = ( 1.1, 3.0 ),
            d0_par2 = ( 1.1, 3.0 ),
            dz_par2 = ( 1.1, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'mixedTripletStepTrkTight',
            preFilterName = 'mixedTripletStepTrkLoose',
            chi2n_par = 0.3,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 4,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 1.0, 4.0 ),
            d0_par2 = ( 1.0, 4.0 ),
            dz_par2 = ( 1.0, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'mixedTripletStepVtx',
            preFilterName = 'mixedTripletStepVtxTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            max_minMissHitOutOrIn = 1,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 1.0, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'mixedTripletStepTrk',
            preFilterName = 'mixedTripletStepTrkTight',
            chi2n_par = 0.25,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 4,
            max_minMissHitOutOrIn = 1,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.8, 4.0 ),
            dz_par2 = ( 0.8, 4.0 )
            )
        ) #end of vpset
    ) #end of clone

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
mixedTripletStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('mixedTripletStepTracks'),
                                   cms.InputTag('mixedTripletStepTracks')),
    hasSelector=cms.vint32(1,1),
    shareFrac=cms.double(0.11),
    indivShareFrac=cms.vdouble(0.11,0.11),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("mixedTripletStepSelector","mixedTripletStepVtx"),
                                       cms.InputTag("mixedTripletStepSelector","mixedTripletStepTrk")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
)                        


MixedTripletStep = cms.Sequence(mixedTripletStepClusters*
                                mixedTripletStepSeedLayersA*
                                mixedTripletStepSeedsA*
                                mixedTripletStepSeedLayersB*
                                mixedTripletStepSeedsB*
                                mixedTripletStepSeeds*
                                mixedTripletStepTrackCandidates*
                                mixedTripletStepTracks*
                                mixedTripletStepSelector*
                                mixedTripletStep)
