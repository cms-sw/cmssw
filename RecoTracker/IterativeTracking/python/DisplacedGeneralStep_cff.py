import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

#----------------------------------------- NEW CLUSTERS (remove previously used clusters)
displacedGeneralStepClusters = _cfg.clusterRemoverForIter("DisplacedGeneralStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(displacedGeneralStepClusters, _cfg.clusterRemoverForIter("DisplacedGeneralStep", _eraName, _postfix))

#----------------------------------------- SEEDING LAYERS
import RecoTracker.TkSeedingLayers.DisplacedGeneralLayerTriplet_cfi
displacedGeneralStepSeedLayers = RecoTracker.TkSeedingLayers.DisplacedGeneralLayerTriplet_cfi.DisplacedGeneralLayerTriplet.clone()


#----------------------------------------- TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegion_cfi import globalTrackingRegion as _globalTrackingRegion
displacedGeneralStepTrackingRegions = _globalTrackingRegion.clone(
   RegionPSet = dict(
     precise = True,
     useMultipleScattering = True,
     originHalfLength = 55,
     originRadius = 10,
     ptMin = 1
   )
)




#----------------------------------------- Triplet seeding

from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import ClusterShapeHitFilterESProducer as _ClusterShapeHitFilterESProducer
displacedGeneralStepClusterShapeHitFilter = _ClusterShapeHitFilterESProducer.clone(
    ComponentName = 'displacedGeneralStepClusterShapeHitFilter',
    doStripShapeCut = cms.bool(False),
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight')
)

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
displacedGeneralStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "displacedGeneralStepSeedLayers",
    trackingRegions = "displacedGeneralStepTrackingRegions",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
)

from RecoTracker.TkSeedGenerator.multiHitFromChi2EDProducer_cfi import multiHitFromChi2EDProducer as _multiHitFromChi2EDProducer
displacedGeneralStepHitTriplets = _multiHitFromChi2EDProducer.clone(
    doublets = "displacedGeneralStepHitDoublets",
    extraPhiKDBox = 0.01,
)


from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
from RecoTracker.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi import StripSubClusterShapeSeedFilter as _StripSubClusterShapeSeedFilter
displacedGeneralStepSeeds = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets = "displacedGeneralStepHitTriplets",
    SeedComparitorPSet = dict(
        ComponentName = 'CombinedSeedComparitor',
        mode = cms.string("and"),
        comparitors = cms.VPSet(
            cms.PSet(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
                ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
                FilterAtHelixStage = cms.bool(True),
                FilterPixelHits = cms.bool(False),
                FilterStripHits = cms.bool(True),
                ClusterShapeHitFilterName = cms.string('displacedGeneralStepClusterShapeHitFilter'),
                ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache") # not really needed here since FilterPixelHits=False
            ), 
            _StripSubClusterShapeSeedFilter.clone()
        )
    )
)



#----------------------------------------- QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_displacedGeneralStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 4,
    minPt = 1,
)

displacedGeneralStepTrajectoryFilter = _displacedGeneralStepTrajectoryFilterBase.clone(
    seedPairPenalty = 1,
)


displacedGeneralStepTrajectoryFilterInOut = displacedGeneralStepTrajectoryFilter.clone()

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
displacedGeneralStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'displacedGeneralStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 10.0,
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
)



#----------------------------------------- TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
displacedGeneralStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'displacedGeneralStepTrajectoryFilter'),
    inOutTrajectoryFilter = dict(refToPSet_ = 'displacedGeneralStepTrajectoryFilterInOut'),
    useSameTrajFilter = False,
    minNrOfHitsForRebuild = 4,
    maxCand = 2,
    estimator = 'displacedGeneralStepChi2Est'
)



#----------------------------------------- MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
displacedGeneralStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'displacedGeneralStepSeeds',
    TrajectoryCleaner = 'displacedGeneralStepTrajectoryCleanerBySharedHits',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    onlyPixelHitsForSeedCleaner = False,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'displacedGeneralStepTrajectoryBuilder'),
    clustersToSkip = 'displacedGeneralStepClusters',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
displacedGeneralStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = 'displacedGeneralStepTrajectoryCleanerBySharedHits',
    fractionShared = 0.25,
    allowSharedFirstHit = True
)





# ----------------------------------------- TRACK FITTING AND SMOOTHING OPTIONS
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
displacedGeneralStepFitterSmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'displacedGeneralStepFitterSmoother',
    EstimateCut = 30,
    MinNumberOfHits = 8,
    Fitter = 'displacedGeneralStepRKFitter',
    Smoother = 'displacedGeneralStepRKSmoother'
)



# Also necessary to specify minimum number of hits after final track fit
displacedGeneralStepRKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = 'displacedGeneralStepRKFitter',
    minHits = 8
)



displacedGeneralStepRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = 'displacedGeneralStepRKSmoother',
    errorRescaling = 10.0,
    minHits = 8
)



import TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi
generalDisplacedFlexibleKFFittingSmoother = TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi.FlexibleKFFittingSmoother.clone(
    ComponentName = 'generalDisplacedFlexibleKFFittingSmoother',
    standardFitter = 'displacedGeneralStepFitterSmoother',
)




import RecoTracker.TrackProducer.TrackProducer_cfi
displacedGeneralStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'displacedGeneralStepTrackCandidates',
    AlgorithmName = 'displacedGeneralStep',
    Fitter = 'generalDisplacedFlexibleKFFittingSmoother',
)


#---------------------------------------- TRACK SELECTION AND QUALITY FLAG SETTING.

from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
displacedGeneralStepClassifier1 = TrackMVAClassifierDetached.clone(
  src = 'displacedGeneralStepTracks',
  mva = dict(GBRForestLabel = 'MVASelectorIter6_13TeV'),
  qualityCuts = [-0.6,-0.45,-0.3]
)
displacedGeneralStepClassifier2 = TrackMVAClassifierPrompt.clone(
src = 'displacedGeneralStepTracks',
  mva = dict(GBRForestLabel = 'MVASelectorIter0_13TeV'),
  qualityCuts = [0.0,0.0,0.0]
)

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
displacedGeneralStep = ClassifierMerger.clone(
  inputClassifiers=['displacedGeneralStepClassifier1','displacedGeneralStepClassifier2']
)

from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toReplaceWith(displacedGeneralStep, displacedGeneralStepClassifier1.clone(
    mva = dict(GBRForestLabel = 'MVASelectorTobTecStep_Phase1'),
    qualityCuts = [-0.6,-0.45,-0.3],
))



DisplacedGeneralStepTask = cms.Task(displacedGeneralStepClusters,
                          displacedGeneralStepSeedLayers,
                          displacedGeneralStepTrackingRegions,
                          displacedGeneralStepHitDoublets,
                          displacedGeneralStepHitTriplets,
                          displacedGeneralStepSeeds,
                          displacedGeneralStepTrackCandidates,
                          displacedGeneralStepTracks,
                          displacedGeneralStepClassifier1,displacedGeneralStepClassifier2,
                          displacedGeneralStep)

DisplacedGeneralStep = cms.Sequence(DisplacedGeneralStepTask)
