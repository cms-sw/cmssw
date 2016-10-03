import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

#######################################################################
# Very large impact parameter tracking using TOB + TEC ring 5 seeding #
#######################################################################

tobTecStepClusters = _cfg.clusterRemoverForIter("TobTecStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(tobTecStepClusters, _cfg.clusterRemoverForIter("TobTecStep", _eraName, _postfix))

# TRIPLET SEEDING LAYERS
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *
tobTecStepSeedLayersTripl = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(
    #TOB
    'TOB1+TOB2+MTOB3','TOB1+TOB2+MTOB4',
    #TOB+MTEC
    'TOB1+TOB2+MTEC1_pos','TOB1+TOB2+MTEC1_neg',
    ),
    TOB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
         skipClusters   = cms.InputTag('tobTecStepClusters')
    ),
    MTOB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         skipClusters   = cms.InputTag('tobTecStepClusters'),
         rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    MTEC = cms.PSet(
        rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('tobTecStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(6),
        maxRing = cms.int32(7)
    )
)
# TRIPLET SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
tobTecStepSeedsTripl = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#OrderedHitsFactory
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.SeedingLayers = 'tobTecStepSeedLayersTripl'
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.ComponentName = 'StandardMultiHitGenerator'
import RecoTracker.TkSeedGenerator.MultiHitGeneratorFromChi2_cfi
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.GeneratorPSet = RecoTracker.TkSeedGenerator.MultiHitGeneratorFromChi2_cfi.MultiHitGeneratorFromChi2.clone(
    extraPhiKDBox = 0.01
    )
#RegionFactory
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.ptMin = 0.55
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.originHalfLength = 20.0
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.originRadius = 3.5
#SeedCreator
tobTecStepSeedsTripl.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsCreator' #empirically better than 'SeedFromConsecutiveHitsTripletOnlyCreator'
tobTecStepSeedsTripl.SeedCreatorPSet.OriginTransverseErrorMultiplier = 1.0
#SeedComparitor
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi

tobTecStepSeedsTripl.SeedComparitorPSet = cms.PSet(
   ComponentName = cms.string('CombinedSeedComparitor'),
   mode = cms.string("and"),
   comparitors = cms.VPSet(
     cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(True),
        ClusterShapeHitFilterName = cms.string('tobTecStepClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache") # not really needed here since FilterPixelHits=False
    ),
    RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi.StripSubClusterShapeSeedFilter.clone()
  )
)
# PAIR SEEDING LAYERS
tobTecStepSeedLayersPair = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TOB1+TEC1_pos','TOB1+TEC1_neg', 
                            'TEC1_pos+TEC2_pos','TEC1_neg+TEC2_neg', 
                            'TEC2_pos+TEC3_pos','TEC2_neg+TEC3_neg', 
                            'TEC3_pos+TEC4_pos','TEC3_neg+TEC4_neg', 
                            'TEC4_pos+TEC5_pos','TEC4_neg+TEC5_neg', 
                            'TEC5_pos+TEC6_pos','TEC5_neg+TEC6_neg', 
                            'TEC6_pos+TEC7_pos','TEC6_neg+TEC7_neg'),
    TOB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
         skipClusters   = cms.InputTag('tobTecStepClusters')
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('tobTecStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)
# PAIR SEEDS
import RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi
tobTecStepClusterShapeHitFilter  = RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi.ClusterShapeHitFilterESProducer.clone(
	ComponentName = cms.string('tobTecStepClusterShapeHitFilter'),
	clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
	doStripShapeCut  = cms.bool(False)
	)

import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
tobTecStepSeedsPair = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
#OrderedHitsFactory
tobTecStepSeedsPair.OrderedHitsFactoryPSet.ComponentName = cms.string('StandardHitPairGenerator')
tobTecStepSeedsPair.OrderedHitsFactoryPSet.SeedingLayers = 'tobTecStepSeedLayersPair'
#RegionFactory
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.ptMin = 0.6
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originHalfLength = 30.0
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originRadius = 6.0
#SeedCreator
tobTecStepSeedsPair.SeedCreatorPSet.OriginTransverseErrorMultiplier = 1.0
#SeedComparitor
tobTecStepSeedsPair.SeedComparitorPSet = cms.PSet(
   ComponentName = cms.string('CombinedSeedComparitor'),
   mode = cms.string("and"),
   comparitors = cms.VPSet(
     cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(True),
        ClusterShapeHitFilterName = cms.string('tobTecStepClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache") # not really needed here since FilterPixelHits=False
    ),
    RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi.StripSubClusterShapeSeedFilter.clone()
  )
)
import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
tobTecStepSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
tobTecStepSeeds.seedCollections = cms.VInputTag(cms.InputTag('tobTecStepSeedsTripl'),cms.InputTag('tobTecStepSeedsPair'))
# LowPU
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(tobTecStepSeeds, RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone(
    OrderedHitsFactoryPSet = dict(SeedingLayers = 'tobTecStepSeedLayers'),
    RegionFactoryPSet = dict(RegionPSet = dict(
        ptMin = 0.6,
        originHalfLength = 30.0,
        originRadius = 6.0,
    ))
))
# Phase1PU70
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toReplaceWith(tobTecStepSeeds, RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone(
    OrderedHitsFactoryPSet = dict(SeedingLayers = 'tobTecStepSeedLayers'),
    RegionFactoryPSet = dict(
        RegionPSet = dict(
            ptMin = 1.0,
            originHalfLength = 15.0,
            originRadius = 2.0
        )
    ),
    SeedCreatorPSet = dict(OriginTransverseErrorMultiplier = 3.0),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(True),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache") # not really needed here since FilterPixelHits=False
    ),
))


# QUALITY CUTS DURING TRACK BUILDING (for inwardss and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_tobTecStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 5,
    minPt = 0.1,
    minHitsMinPt = 3
    )
tobTecStepTrajectoryFilter = _tobTecStepTrajectoryFilterBase.clone(
    seedPairPenalty = 1,
)
trackingLowPU.toReplaceWith(tobTecStepTrajectoryFilter, _tobTecStepTrajectoryFilterBase.clone(
    minimumNumberOfHits = 6,
))
trackingPhase1PU70.toReplaceWith(tobTecStepTrajectoryFilter, _tobTecStepTrajectoryFilterBase.clone(
    minimumNumberOfHits = 6,
))

tobTecStepInOutTrajectoryFilter = tobTecStepTrajectoryFilter.clone(
    minimumNumberOfHits = 4,
)


import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
tobTecStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('tobTecStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(16.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))
)
trackingLowPU.toModify(tobTecStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')
)
trackingPhase1PU70.toModify(tobTecStepChi2Est,
    MaxChi2 = 9.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
tobTecStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('tobTecStepTrajectoryFilter')),
    inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('tobTecStepInOutTrajectoryFilter')),
    useSameTrajFilter = False,
    minNrOfHitsForRebuild = 4,
    alwaysUseInvalidHits = False,
    maxCand = 2,
    estimator = cms.string('tobTecStepChi2Est'),
    #startSeedHitsInRebuild = True
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )
# Important note for LowPU: in RunI_TobTecStep the
# inOutTrajectoryFilter parameter is spelled as
# inOutTrajectoryFilterName, and I suspect it has no effect there. I
# chose to "fix" the behaviour here, so the era is not fully
# equivalent to the customize. To restore the customize behaviour,
# uncomment the following lines
#trackingLowPU.toModify(tobTecStepTrajectoryBuilder,
#    inOutTrajectoryFilter = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.inOutTrajectoryFilter.clone(),
#    inOutTrajectoryFilterName = cms.PSet(refToPSet_ = cms.string('tobTecStepInOutTrajectoryFilter'))
#)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
tobTecStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('tobTecStepSeeds'),
    clustersToSkip = cms.InputTag('tobTecStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('tobTecStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
tobTecStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = cms.string('tobTecStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.09),
    allowSharedFirstHit = cms.bool(True)
    )
tobTecStepTrackCandidates.TrajectoryCleaner = 'tobTecStepTrajectoryCleanerBySharedHits'
trackingLowPU.toModify(tobTecStepTrajectoryCleanerBySharedHits, fractionShared = 0.19)
trackingPhase1PU70.toModify(tobTecStepTrajectoryCleanerBySharedHits, fractionShared = 0.08)

# TRACK FITTING AND SMOOTHING OPTIONS
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
tobTecStepFitterSmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'tobTecStepFitterSmoother',
    EstimateCut = 30,
    MinNumberOfHits = 7,
    Fitter = cms.string('tobTecStepRKFitter'),
    Smoother = cms.string('tobTecStepRKSmoother')
    )
trackingLowPU.toModify(tobTecStepFitterSmoother, MinNumberOfHits = 8)
trackingPhase1PU70.toModify(tobTecStepFitterSmoother, MinNumberOfHits = 8)

tobTecStepFitterSmootherForLoopers = tobTecStepFitterSmoother.clone(
    ComponentName = 'tobTecStepFitterSmootherForLoopers',
    Fitter = cms.string('tobTecStepRKFitterForLoopers'),
    Smoother = cms.string('tobTecStepRKSmootherForLoopers')
)

# Also necessary to specify minimum number of hits after final track fit
tobTecStepRKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('tobTecStepRKFitter'),
    minHits = 7
)
trackingLowPU.toModify(tobTecStepRKTrajectoryFitter, minHits = 8)
trackingPhase1PU70.toModify(tobTecStepRKTrajectoryFitter, minHits = 8)

tobTecStepRKTrajectoryFitterForLoopers = tobTecStepRKTrajectoryFitter.clone(
    ComponentName = cms.string('tobTecStepRKFitterForLoopers'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
)

tobTecStepRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('tobTecStepRKSmoother'),
    errorRescaling = 10.0,
    minHits = 7
)
trackingLowPU.toModify(tobTecStepRKTrajectorySmoother, minHits = 8)
trackingPhase1PU70.toModify(tobTecStepRKTrajectorySmoother, minHits = 8)

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
    AlgorithmName = cms.string('tobTecStep'),
    #Fitter = 'tobTecStepFitterSmoother',
    Fitter = 'tobTecFlexibleKFFittingSmoother',
    )



# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
tobTecStepClassifier1 = TrackMVAClassifierDetached.clone()
tobTecStepClassifier1.src = 'tobTecStepTracks'
tobTecStepClassifier1.GBRForestLabel = 'MVASelectorIter6_13TeV'
tobTecStepClassifier1.qualityCuts = [-0.6,-0.45,-0.3]
tobTecStepClassifier2 = TrackMVAClassifierPrompt.clone()
tobTecStepClassifier2.src = 'tobTecStepTracks'
tobTecStepClassifier2.GBRForestLabel = 'MVASelectorIter0_13TeV'
tobTecStepClassifier2.qualityCuts = [0.0,0.0,0.0]

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
tobTecStep = ClassifierMerger.clone()
tobTecStep.inputClassifiers=['tobTecStepClassifier1','tobTecStepClassifier2']

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
trackingLowPU.toReplaceWith(tobTecStep, RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'tobTecStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter6'),
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'tobTecStepLoose',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 2,
            d0_par1 = ( 2.0, 4.0 ),
            dz_par1 = ( 1.8, 4.0 ),
            d0_par2 = ( 2.0, 4.0 ),
            dz_par2 = ( 1.8, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'tobTecStepTight',
            preFilterName = 'tobTecStepLoose',
            chi2n_par = 0.3,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 2,
            d0_par1 = ( 1.5, 4.0 ),
            dz_par1 = ( 1.4, 4.0 ),
            d0_par2 = ( 1.5, 4.0 ),
            dz_par2 = ( 1.4, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'tobTecStepTight',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 2,
            d0_par1 = ( 1.4, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.4, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
        ),
    ] #end of vpset
)) #end of clone



TobTecStep = cms.Sequence(tobTecStepClusters*
                          tobTecStepSeedLayersTripl*
                          tobTecStepSeedsTripl*
                          tobTecStepSeedLayersPair*
                          tobTecStepSeedsPair*
                          tobTecStepSeeds*
                          tobTecStepTrackCandidates*
                          tobTecStepTracks*
                          tobTecStepClassifier1*tobTecStepClassifier2*
                          tobTecStep)



### Following are specific for LowPU and/or Phase1PU70, they're collected here to
### not to interfere too much with the default configuration
# For Phase1PU70
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
tobTecStepSeedClusters = _trackClusterRemover.clone(
    maxChi2                                  = 9.0,
    trajectories                             = "mixedTripletStepTracks",
    pixelClusters                            = "siPixelClusters",
    stripClusters                            = "siStripClusters",
    oldClusterRemovalInfo                    = "mixedTripletStepClusters",
    overrideTrkQuals                         = 'mixedTripletStep',
    TrackQuality                             = 'highPurity',
    minNumberOfLayersWithMeasBeforeFiltering = 0,
)

# SEEDING LAYERS
tobTecStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
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
        skipClusters = cms.InputTag('tobTecStepClusters'),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny'))
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('tobTecStepClusters'),
        #    untracked bool useSimpleRphiHitsCleaner = false
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)
trackingPhase1PU70.toModify(tobTecStepSeedLayers,
    TOB = dict(
        skipClusters = 'tobTecStepSeedClusters',
        clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
    ),
    TEC = dict(
        skipClusters = 'tobTecStepSeedClusters',
        clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
    ),
)

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
tobTecStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'tobTecStepTracks',
    trackSelectors = [
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
    ] #end of vpset
) #end of clone

trackingLowPU.toReplaceWith(TobTecStep, cms.Sequence(
    tobTecStepClusters*
    tobTecStepSeedLayers*
    tobTecStepSeeds*
    tobTecStepTrackCandidates*
    tobTecStepTracks*
    tobTecStep
))
trackingPhase1PU70.toReplaceWith(TobTecStep, cms.Sequence(
    tobTecStepClusters*
    tobTecStepSeedClusters*
    tobTecStepSeedLayers*
    tobTecStepSeeds*
    tobTecStepTrackCandidates*
    tobTecStepTracks*
    tobTecStepSelector
))
