import FWCore.ParameterSet.Config as cms

#######################################################################
# Very large impact parameter tracking using TOB + TEC ring 5 seeding #
#######################################################################

tobTecStepClusters = cms.EDProducer("TrackClusterRemover",
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

tobTecStepSeedClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution = cms.bool(True),
    oldClusterRemovalInfo = cms.InputTag("mixedTripletStepClusters"),
    trajectories = cms.InputTag("mixedTripletStepTracks"),
    overrideTrkQuals = cms.InputTag('mixedTripletStep'),
    TrackQuality = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    doStripChargeCheck = cms.bool(True),
    stripRecHits = cms.string('siStripMatchedRecHits'),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0),
        minGoodStripCharge = cms.double(80.0)
    )
)

# SEEDING LAYERS
tobTecStepSeedLayers = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('tobTecStepSeedLayers'),

    layerList = cms.vstring('TOB1+TOB2', 
        'TOB1+TEC1_pos', 'TOB1+TEC1_neg', 
        'TEC1_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 'TEC4_pos+TEC5_pos', 
        'TEC5_pos+TEC6_pos', 'TEC6_pos+TEC7_pos', 
        'TEC1_neg+TEC2_neg', 'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 'TEC4_neg+TEC5_neg', 
        'TEC5_neg+TEC6_neg', 'TEC6_neg+TEC7_neg'),
    TOB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('tobTecStepSeedClusters'),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('tobTecStepSeedClusters'),
        #    untracked bool useSimpleRphiHitsCleaner = false
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)
# SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
tobTecStepSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
tobTecStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'tobTecStepSeedLayers'
tobTecStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 1.0
tobTecStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
tobTecStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 2.0
tobTecStepSeeds.SeedCreatorPSet.OriginTransverseErrorMultiplier = 3.0

tobTecStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(True),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
    )
tobTecStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
tobTecStepSeeds.OrderedHitsFactoryPSet.maxElement = cms.uint32(0)

# QUALITY CUTS DURING TRACK BUILDING (for inwardss and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

tobTecStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'tobTecStepTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 6,
    minPt = 0.1,
    minHitsMinPt = 3
    )
    )

tobTecStepInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'tobTecStepInOutTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 4,
    minPt = 0.1,
    minHitsMinPt = 3
    )
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
tobTecStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('tobTecStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
tobTecStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'tobTecStepTrajectoryBuilder',
    MeasurementTrackerName = '',
    clustersToSkip = cms.InputTag('tobTecStepClusters'),
    trajectoryFilterName = 'tobTecStepTrajectoryFilter',
    inOutTrajectoryFilterName = 'tobTecStepInOutTrajectoryFilter',
    useSameTrajFilter = False,
    minNrOfHitsForRebuild = 4,
    alwaysUseInvalidHits = False,
    maxCand = 2,
    estimator = cms.string('tobTecStepChi2Est'),
    #startSeedHitsInRebuild = True
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)  
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
tobTecStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('tobTecStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = 'tobTecStepTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
tobTecStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = cms.string('tobTecStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.08),
    allowSharedFirstHit = cms.bool(True)
    )
tobTecStepTrackCandidates.TrajectoryCleaner = 'tobTecStepTrajectoryCleanerBySharedHits'

# TRACK FITTING AND SMOOTHING OPTIONS
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
tobTecStepFitterSmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'tobTecStepFitterSmoother',
    EstimateCut = 30,
    MinNumberOfHits = 8,
    Fitter = cms.string('tobTecStepRKFitter'),
    Smoother = cms.string('tobTecStepRKSmoother')
    )

tobTecStepFitterSmootherForLoopers = tobTecStepFitterSmoother.clone(
    ComponentName = 'tobTecStepFitterSmootherForLoopers',
    Fitter = cms.string('tobTecStepRKFitterForLoopers'),
    Smoother = cms.string('tobTecStepRKSmootherForLoopers')
)

# Also necessary to specify minimum number of hits after final track fit
tobTecStepRKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('tobTecStepRKFitter'),
    minHits = 8
)
tobTecStepRKTrajectoryFitterForLoopers = tobTecStepRKTrajectoryFitter.clone(
    ComponentName = cms.string('tobTecStepRKFitterForLoopers'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
)

tobTecStepRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('tobTecStepRKSmoother'),
    errorRescaling = 10.0,
    minHits = 8
)
tobTecStepRKTrajectorySmootherForLoopers = tobTecStepRKTrajectorySmoother.clone(
    ComponentName = cms.string('tobTecStepRKSmootherForLoopers'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
)

import TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi
tobTecFlexibleKFFittingSmoother = TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi.FlexibleKFFittingSmoother.clone(
    ComponentName = cms.string('tobTecFlexibleKFFittingSmoother'),
    standardFitter = cms.string('tobTecStepFitterSmoother'),
    looperFitter = cms.string('tobTecStepFitterSmootherForLoopers'),
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
tobTecStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'tobTecStepTrackCandidates',
    AlgorithmName = cms.string('iter7'),
    #Fitter = 'tobTecStepFitterSmoother',
    Fitter = cms.string('tobTecFlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle')
    )

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
tobTecStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='tobTecStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'tobTecStepLoose',
            chi2n_par = 0.25,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 2,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.2, 4.0 ),
            d0_par2 = ( 1.3, 4.0 ),
            dz_par2 = ( 1.2, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'tobTecStepTight',
            preFilterName = 'tobTecStepLoose',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 2,
            max_minMissHitOutOrIn = 1,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.0, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.0, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'tobTecStep',
            preFilterName = 'tobTecStepTight',
            chi2n_par = 0.15,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 6,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 2,
            max_minMissHitOutOrIn = 0,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.8, 4.0 )
            ),
        ) #end of vpset
    ) #end of clone


TobTecStep = cms.Sequence(tobTecStepClusters*
                          tobTecStepSeedClusters*
                          tobTecStepSeeds*
                          tobTecStepTrackCandidates*
                          tobTecStepTracks*
                          tobTecStepSelector)
