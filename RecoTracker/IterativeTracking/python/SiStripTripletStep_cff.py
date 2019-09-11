import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

#----------------------------------------- NEW CLUSTERS (remove previously used clusters)
siStripTripletStepClusters = _cfg.clusterRemoverForIter("SiStripTripletStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(siStripTripletStepClusters, _cfg.clusterRemoverForIter("SiStripTripletStep", _eraName, _postfix))

#----------------------------------------- SEEDING LAYERS
import RecoTracker.TkSeedingLayers.SiStripLayerTriplets_cfi
siStripTripletStepSeedLayers = RecoTracker.TkSeedingLayers.SiStripLayerTriplets_cfi.SiStripLayerTriplets.clone()


#----------------------------------------- TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegion_cfi import globalTrackingRegion as _globalTrackingRegion
#from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import GlobalTrackingRegion as _globalTrackingRegion
siStripTripletStepTrackingRegions = _globalTrackingRegion.clone(
   RegionPSet = dict(
     precise = cms.bool(True),
     useMultipleScattering = cms.bool(False),
     originHalfLength = cms.double(100),
     originRadius = cms.double(100)
   )
)




#----------------------------------------- Triplet seeding

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import ClusterShapeHitFilterESProducer as _ClusterShapeHitFilterESProducer
siStripTripletStepClusterShapeHitFilter = _ClusterShapeHitFilterESProducer.clone(
    ComponentName = 'siStripTripletStepClusterShapeHitFilter',
    doStripShapeCut = cms.bool(False),
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight')
)

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
siStripTripletStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "siStripTripletStepSeedLayers",
    trackingRegions = "siStripTripletStepTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)

from RecoTracker.TkSeedGenerator.multiHitFromChi2EDProducer_cfi import multiHitFromChi2EDProducer as _multiHitFromChi2EDProducer
siStripTripletStepHitTriplets = _multiHitFromChi2EDProducer.clone(
    doublets = "siStripTripletStepHitDoublets",
    extraPhiKDBox = 0.01,
)


#from RecoPixelVertexing.PixelTriplets.pixelTripletLargeTipEDProducer_cfi import pixelTripletLargeTipEDProducer as _pixelTripletLargeTipEDProducer
#from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
#siStripTripletStepHitTriplets = _pixelTripletHLTEDProducer.clone(
#    doublets = "siStripTripletStepHitDoublets",
#    produceSeedingHitSets = True
#)
#
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi import StripSubClusterShapeSeedFilter as _StripSubClusterShapeSeedFilter


_siStripTripletStepSeedComparitorPSet = dict(
    ComponentName = 'CombinedSeedComparitor',
    mode = cms.string("and"),
    comparitors = cms.VPSet(
        cms.PSet(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
            ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
            FilterAtHelixStage = cms.bool(True),
            FilterPixelHits = cms.bool(False),
            FilterStripHits = cms.bool(True),
            ClusterShapeHitFilterName = cms.string('siStripTripletStepClusterShapeHitFilter'),
            ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache") # not really needed here since FilterPixelHits=False
        ),
        _StripSubClusterShapeSeedFilter.clone()
    )
)

siStripTripletStepSeeds = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(#empirically better than 'SeedFromConsecutiveHitsTripletOnlyCreator'
    seedingHitSets = "siStripTripletStepHitTriplets",
    SeedComparitorPSet = _siStripTripletStepSeedComparitorPSet,
)









#----------------------------------------- QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_siStripTripletStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 3,
    minPt = 0.1,
)

siStripTripletStepTrajectoryFilter = _siStripTripletStepTrajectoryFilterBase.clone(
    seedPairPenalty = 1,
)


siStripTripletStepTrajectoryFilterInOut = siStripTripletStepTrajectoryFilter.clone(
    minimumNumberOfHits = 4,
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
siStripTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('siStripTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(20.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
)



#----------------------------------------- TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
siStripTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('siStripTripletStepTrajectoryFilter')),
    inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('siStripTripletStepTrajectoryFilterInOut')),
    useSameTrajFilter = False,
    minNrOfHitsForRebuild = 4,
    #maxCand = 4,
    estimator = cms.string('siStripTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
    )



#----------------------------------------- MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
siStripTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('siStripTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('siStripTripletStepTrajectoryBuilder')),
    #clustersToSkip = cms.InputTag('siStripTripletStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
siStripTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = cms.string('siStripTripletStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.19),
    allowSharedFirstHit = cms.bool(True)
    )
siStripTripletStepTrackCandidates.TrajectoryCleaner = 'siStripTripletStepTrajectoryCleanerBySharedHits'





# ----------------------------------------- TRACK FITTING AND SMOOTHING OPTIONS
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
siStripTripletStepFitterSmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'siStripTripletStepFitterSmoother',
    EstimateCut = 30,
    MinNumberOfHits = 7,
    Fitter = cms.string('siStripTripletStepRKFitter'),
    Smoother = cms.string('siStripTripletStepRKSmoother')
    )


siStripTripletStepFitterSmootherForLoopers = siStripTripletStepFitterSmoother.clone(
    ComponentName = 'siStripTripletStepFitterSmootherForLoopers',
    Fitter = cms.string('siStripTripletStepRKFitterForLoopers'),
    Smoother = cms.string('siStripTripletStepRKSmootherForLoopers')
)

# Also necessary to specify minimum number of hits after final track fit
siStripTripletStepRKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('siStripTripletStepRKFitter'),
    minHits = 7
)


siStripTripletStepRKTrajectoryFitterForLoopers = siStripTripletStepRKTrajectoryFitter.clone(
    ComponentName = cms.string('siStripTripletStepRKFitterForLoopers'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
)

siStripTripletStepRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('siStripTripletStepRKSmoother'),
    errorRescaling = 10.0,
    minHits = 7
)


siStripTripletStepRKTrajectorySmootherForLoopers = siStripTripletStepRKTrajectorySmoother.clone(
    ComponentName = cms.string('siStripTripletStepRKSmootherForLoopers'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
)

import TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi
siStripTripletFlexibleKFFittingSmoother = TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi.FlexibleKFFittingSmoother.clone(
    ComponentName = cms.string('siStripTripletFlexibleKFFittingSmoother'),
    standardFitter = cms.string('siStripTripletStepFitterSmoother'),
    looperFitter = cms.string('siStripTripletStepFitterSmootherForLoopers'),
)


#----------------------------------------- TRACK FITTING
#import RecoTracker.TrackProducer.TrackProducer_cfi
#siStripTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
#    src = 'siStripTripletStepTrackCandidates',
#    AlgorithmName = cms.string('siStripTripletStep'),
#    Fitter = cms.string('FlexibleKFFittingSmoother')
#    )
#
#from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
#siStripTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
#        ComponentName = cms.string('siStripTripletStepTrajectoryCleanerBySharedHits'),
#            fractionShared = cms.double(0.16),
#            allowSharedFirstHit = cms.bool(True)
#            )
#siStripTripletStepTrackCandidates.TrajectoryCleaner = 'siStripTripletStepTrajectoryCleanerBySharedHits'

import RecoTracker.TrackProducer.TrackProducer_cfi
siStripTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'siStripTripletStepTrackCandidates',
    AlgorithmName = cms.string('siStripTripletStep'),
    #Fitter = 'siStripTripletStepFitterSmoother',
    Fitter = 'siStripTripletFlexibleKFFittingSmoother',
    )



#----------------------------------------- TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
siStripTripletStepClassifier1 = TrackMVAClassifierDetached.clone()
siStripTripletStepClassifier1.src = 'siStripTripletStepTracks'
siStripTripletStepClassifier1.GBRForestLabel = 'MVASelectorIter6_13TeV'
siStripTripletStepClassifier1.qualityCuts = [-0.6,-0.45,-0.3]
siStripTripletStepClassifier2 = TrackMVAClassifierPrompt.clone()
siStripTripletStepClassifier2.src = 'siStripTripletStepTracks'
siStripTripletStepClassifier2.GBRForestLabel = 'MVASelectorIter0_13TeV'
siStripTripletStepClassifier2.qualityCuts = [0.0,0.0,0.0]

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
siStripTripletStep = ClassifierMerger.clone()
siStripTripletStep.inputClassifiers=['siStripTripletStepClassifier1','siStripTripletStepClassifier2']

from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1.toReplaceWith(siStripTripletStep, siStripTripletStepClassifier1.clone(
     GBRForestLabel = 'MVASelectorTobTecStep_Phase1',
     qualityCuts = [-0.6,-0.45,-0.3],
))
trackingPhase1QuadProp.toReplaceWith(siStripTripletStep, siStripTripletStepClassifier1.clone(
     GBRForestLabel = 'MVASelectorTobTecStep_Phase1',
     qualityCuts = [-0.6,-0.45,-0.3],
))





SiStripTripletStep = cms.Sequence(siStripTripletStepClusters*
                          siStripTripletStepSeedLayers*
                          siStripTripletStepTrackingRegions*
                          siStripTripletStepHitDoublets*
                          siStripTripletStepHitTriplets*
                          siStripTripletStepSeeds*
                          #siStripTripletStepSeedLayersPair*
                          #siStripTripletStepTrackingRegionsPair*
                          #siStripTripletStepHitDoubletsPair*
                          #siStripTripletStepSeedsPair*
                          siStripTripletStepSeeds*
                          siStripTripletStepTrackCandidates*
                          siStripTripletStepTracks*
                          siStripTripletStepClassifier1*siStripTripletStepClassifier2*
                          siStripTripletStep)

