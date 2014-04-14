import FWCore.ParameterSet.Config as cms


# NEW CLUSTERS (remove previously used clusters)
#jetCoreRegionalStepClusters = cms.EDProducer("TrackClusterRemover",
#    clusterLessSolution = cms.bool(True),
#    oldClusterRemovalInfo = cms.InputTag("lowPtTripletStepClusters"),
#    trajectories = cms.InputTag("lowPtTripletStepTracks"),
#    overrideTrkQuals = cms.InputTag('lowPtTripletStepSelector','lowPtTripletStep'),
#    TrackQuality = cms.string('highPurity'),
#    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
#    pixelClusters = cms.InputTag("siPixelClusters"),
#    stripClusters = cms.InputTag("siStripClusters"),
#    Common = cms.PSet(
#        maxChi2 = cms.double(9.0)
#    )
#)

# SEEDING LAYERS
jetCoreRegionalStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
        'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
        'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
	#'BPix2+TIB1',
	'BPix3+TIB1',
	#'BPix2+TIB2',
	'BPix3+TIB2'),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits'),
    #    skipClusters = cms.InputTag('jetCoreRegionalStepClusters')
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits'),
     #   skipClusters = cms.InputTag('jetCoreRegionalStepClusters')
    )
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff
jetCoreRegionalStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
jetsForCoreTracking = cms.EDFilter("CandPtrSelector", src = cms.InputTag("ak5CaloJets"), cut = cms.string("pt > 100 && abs(eta) < 2.5"))
jetCoreRegionalStepSeeds.RegionFactoryPSet = cms.PSet(
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet(
        precise = cms.bool( True ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 10. ),
        originHalfLength = cms.double( 0.2 ),
        deltaPhiRegion = cms.double( 0.10 ), 
        deltaEtaRegion = cms.double( 0.10 ), 
        JetSrc = cms.InputTag( "jetsForCoreTracking" ),
        vertexSrc = cms.InputTag( "firstStepGoodPrimaryVertices" ),
      ))

jetCoreRegionalStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'jetCoreRegionalStepSeedLayers'

jetCoreRegionalStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none'),
#PixelClusterShapeSeedComparitor'),
#        FilterAtHelixStage = cms.bool(True),
#        FilterPixelHits = cms.bool(True),
#        FilterStripHits = cms.bool(False),
 #       ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
jetCoreRegionalStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'jetCoreRegionalStepTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    minimumNumberOfHits = 3,
    minPt = 0.1
    )
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
jetCoreRegionalStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('jetCoreRegionalStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
jetCoreRegionalStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'jetCoreRegionalStepTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'jetCoreRegionalStepTrajectoryFilter',
#    clustersToSkip = cms.InputTag('jetCoreRegionalStepClusters'),
    maxCand = 50,
    estimator = cms.string('jetCoreRegionalStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7),
#    bestHitOnly=False
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
jetCoreRegionalStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('jetCoreRegionalStepSeeds'),
    TrajectoryBuilder = 'jetCoreRegionalStepTrajectoryBuilder',
    maxSeedsBeforeCleaning = cms.uint32(10000),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    #numHitsForSeedCleaner = cms.int32(50),
    #onlyPixelHitsForSeedCleaner = cms.bool(True),

)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
jetCoreRegionalStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter7'),
    src = 'jetCoreRegionalStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
import RecoTracker.IterativeTracking.LowPtTripletStep_cff
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
jetCoreRegionalStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='jetCoreRegionalStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'jetCoreRegionalStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'jetCoreRegionalStepTight',
            preFilterName = 'jetCoreRegionalStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'jetCoreRegionalStep',
            preFilterName = 'jetCoreRegionalStepTight',
            ),
        ) #end of vpset
    ) #end of clone

import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
firstStepPrimaryVertices=RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
firstStepPrimaryVertices.TrackLabel = cms.InputTag("initialStepTracks")

firstStepGoodPrimaryVertices = cms.EDFilter(
     "PrimaryVertexObjectFilter",
     filterParams = cms.PSet(
	     #pvSrc = cms.InputTag('firstStepPrimaryVertices'),#try to remove this one
     	     minNdof = cms.double(25.0),
             maxZ = cms.double(15.0),
             maxRho = cms.double(2.0)
     ),
     src=cms.InputTag('firstStepPrimaryVertices')#why need twice the same src?
   )

# Final sequence
JetCoreRegionalStep = cms.Sequence(jetsForCoreTracking*
                                   firstStepPrimaryVertices*
                                   firstStepGoodPrimaryVertices*
                                   #jetCoreRegionalStepClusters*
                                   jetCoreRegionalStepSeedLayers*
                                   jetCoreRegionalStepSeeds*
                                   jetCoreRegionalStepTrackCandidates*
                                   jetCoreRegionalStepTracks*
                                   jetCoreRegionalStepSelector)
