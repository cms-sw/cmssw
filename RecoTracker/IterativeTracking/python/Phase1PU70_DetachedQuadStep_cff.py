import FWCore.ParameterSet.Config as cms

###############################################
# Low pT and detached tracks from pixel triplets
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS

detachedQuadStepClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution = cms.bool(True),
    oldClusterRemovalInfo = cms.InputTag("lowPtTripletStepClusters"),
    trajectories = cms.InputTag("lowPtTripletStepTracks"),
    overrideTrkQuals = cms.InputTag('lowPtTripletStepSelector','lowPtTripletStep'),
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
detachedQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone(
    ComponentName = cms.string('detachedQuadStepSeedLayers'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
                            'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
                            'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
                            'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                            'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                            'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                            'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg')
    )
detachedQuadStepSeedLayers.BPix.skipClusters = cms.InputTag('detachedQuadStepClusters')
detachedQuadStepSeedLayers.FPix.skipClusters = cms.InputTag('detachedQuadStepClusters')

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
detachedQuadStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            ptMin = 0.3,
            originRadius = 0.5,
            nSigmaZ = 4.0
            )
    ),
    SeedMergerPSet = cms.PSet(
        layerListName = cms.string('PixelSeedMergerQuadruplets'),
	addRemainingTriplets = cms.bool(False),
	mergeTriplets = cms.bool(True),
	ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    )
)
detachedQuadStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'detachedQuadStepSeedLayers'
detachedQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
detachedQuadStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
detachedQuadStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
    )
detachedQuadStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
detachedQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
detachedQuadStepStandardTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'detachedQuadStepStandardTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHitsFraction = cms.double(1./10.),
    constantValueForLostHitsFractionFilter = cms.double(0.501),
    minimumNumberOfHits = 3,
    minPt = 0.075
    )
    )

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilterESProducer_cfi import *
# Composite filter
import TrackingTools.TrajectoryFiltering.CompositeTrajectoryFilterESProducer_cfi
detachedQuadStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.CompositeTrajectoryFilterESProducer_cfi.compositeTrajectoryFilterESProducer.clone(
    ComponentName = cms.string('detachedQuadStepTrajectoryFilter'),
    filterNames   = cms.vstring('detachedQuadStepStandardTrajectoryFilter',
                                'clusterShapeTrajectoryFilter')
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
detachedQuadStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('detachedQuadStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
detachedQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'detachedQuadStepTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'detachedQuadStepTrajectoryFilter',
    clustersToSkip = cms.InputTag('detachedQuadStepClusters'),
    maxCand = 2,
    alwaysUseInvalidHits = False,
    estimator = cms.string('detachedQuadStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
detachedQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('detachedQuadStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = 'detachedQuadStepTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
detachedQuadStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('detachedQuadStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.095),
            allowSharedFirstHit = cms.bool(True)
            )
detachedQuadStepTrackCandidates.TrajectoryCleaner = 'detachedQuadStepTrajectoryCleanerBySharedHits'

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
detachedQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter4'),
    src = 'detachedQuadStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle')
    )

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedQuadStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='detachedQuadStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedQuadStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedQuadStepTrkLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.3, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedQuadStepVtxTight',
            preFilterName = 'detachedQuadStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 0.9, 3.0 ),
            dz_par2 = ( 0.9, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedQuadStepTrkTight',
            preFilterName = 'detachedQuadStepTrkLoose',
            chi2n_par = 0.35,
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
            name = 'detachedQuadStepVtx',
            preFilterName = 'detachedQuadStepVtxTight',
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
            name = 'detachedQuadStepTrk',
            preFilterName = 'detachedQuadStepTrkTight',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 4,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.8, 4.0 ),
            dz_par2 = ( 0.8, 4.0 )
            )
        ) #end of vpset
    ) #end of clone

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
detachedQuadStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('detachedQuadStepTracks'),
                                   cms.InputTag('detachedQuadStepTracks')),
    hasSelector=cms.vint32(1,1),
    shareFrac = cms.double(0.095),
    indivShareFrac=cms.vdouble(0.095,0.095),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("detachedQuadStepSelector","detachedQuadStepVtx"),
                                       cms.InputTag("detachedQuadStepSelector","detachedQuadStepTrk")),
    setsToMerge = cms.VPSet(cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
)

DetachedQuadStep = cms.Sequence(detachedQuadStepClusters*
                                               detachedQuadStepSeeds*
                                               detachedQuadStepTrackCandidates*
                                               detachedQuadStepTracks*
                                               detachedQuadStepSelector*
                                               detachedQuadStep)
