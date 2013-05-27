import FWCore.ParameterSet.Config as cms

###############################################
# Low pT and detached tracks from pixel triplets
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS

detachedTripletStepClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution = cms.bool(True),
    oldClusterRemovalInfo = cms.InputTag("pixelPairStepClusters"),
    trajectories = cms.InputTag("pixelPairStepTracks"),
    overrideTrkQuals = cms.InputTag('pixelPairStepSelector','pixelPairStep'),
    TrackQuality = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    )
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
detachedTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone(
    ComponentName = 'detachedTripletStepSeedLayers'
    )
detachedTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('detachedTripletStepClusters')
detachedTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('detachedTripletStepClusters')

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
detachedTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
detachedTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'detachedTripletStepSeedLayers'
detachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
detachedTripletStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.5
detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.0

detachedTripletStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
detachedTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'detachedTripletStepTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHitsFraction = cms.double(1./10.),
    constantValueForLostHitsFractionFilter = cms.double(0.601),
    minimumNumberOfHits = 3,
    minPt = 0.075
    )
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
detachedTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('detachedTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
detachedTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'detachedTripletStepTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'detachedTripletStepTrajectoryFilter',
    clustersToSkip = cms.InputTag('detachedTripletStepClusters'),
    maxCand = 2,
    alwaysUseInvalidHits = False,
    estimator = cms.string('detachedTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
detachedTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('detachedTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = 'detachedTripletStepTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
detachedTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('detachedTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.13),
            allowSharedFirstHit = cms.bool(True)
            )
detachedTripletStepTrackCandidates.TrajectoryCleaner = 'detachedTripletStepTrajectoryCleanerBySharedHits'


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
detachedTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter3'),
    src = 'detachedTripletStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

# TRACK SELECTION AND QUALITY FLAG SETTING.

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='detachedTripletStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedTripletStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedTripletStepTrkLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.4, 4.0 ),
            d0_par2 = ( 1.4, 4.0 ),
            dz_par2 = ( 1.4, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedTripletStepVtxTight',
            preFilterName = 'detachedTripletStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.95, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedTripletStepTrkTight',
            preFilterName = 'detachedTripletStepTrkLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 4,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedTripletStepVtx',
            preFilterName = 'detachedTripletStepVtxTight',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 3.0 ),
            dz_par1 = ( 0.8, 3.0 ),
            d0_par2 = ( 0.8, 3.0 ),
            dz_par2 = ( 0.8, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedTripletStepTrk',
            preFilterName = 'detachedTripletStepTrkTight',
            chi2n_par = 0.25,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 4,
            d0_par1 = ( 1.0, 4.0 ),
            dz_par1 = ( 1.0, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
            )
        ) #end of vpset
    ) #end of clone


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
detachedTripletStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('detachedTripletStepTracks'),
                                   cms.InputTag('detachedTripletStepTracks')),
    hasSelector=cms.vint32(1,1),
    shareFrac = cms.double(0.13),
    indivShareFrac=cms.vdouble(0.13,0.13),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("detachedTripletStepSelector","detachedTripletStepVtx"),
                                       cms.InputTag("detachedTripletStepSelector","detachedTripletStepTrk")),
    setsToMerge = cms.VPSet(cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
)

DetachedTripletStep = cms.Sequence(detachedTripletStepClusters*
                                               detachedTripletStepSeeds*
                                               detachedTripletStepTrackCandidates*
                                               detachedTripletStepTracks*
                                               detachedTripletStepSelector*
                                               detachedTripletStep)
