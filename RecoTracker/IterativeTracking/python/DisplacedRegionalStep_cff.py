import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

#for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

# for no-loopers
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

#############################################################################
# Tracking seeded by regions of interest targeting exotic physics scenarios #
#############################################################################

displacedRegionalStepClusters = _cfg.clusterRemoverForIter("DisplacedRegionalStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(displacedRegionalStepClusters, _cfg.clusterRemoverForIter("DisplacedRegionalStep", _eraName, _postfix))

# TRIPLET SEEDING LAYERS
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *
import RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi as _mod

displacedRegionalStepSeedLayersTripl = _mod.seedingLayersEDProducer.clone(
    layerList = [
    #TOB
    'TOB1+TOB2+MTOB3',
    #TOB+MTEC
    'TOB1+TOB2+MTEC1_pos','TOB1+TOB2+MTEC1_neg',
    ],
    TOB = dict(
         TTRHBuilder      = cms.string('WithTrackAngle'),
         clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         matchedRecHits   = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
         skipClusters     = cms.InputTag('displacedRegionalStepClusters')
    ),
    MTOB = dict(
         TTRHBuilder      = cms.string('WithTrackAngle'),
         clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         skipClusters     = cms.InputTag('displacedRegionalStepClusters'),
         rphiRecHits      = cms.InputTag('siStripMatchedRecHits','rphiRecHit')
    ),
    MTEC = dict(
        rphiRecHits      = cms.InputTag('siStripMatchedRecHits','rphiRecHit'),
        skipClusters     = cms.InputTag('displacedRegionalStepClusters'),
        useRingSlector   = cms.bool(True),
        TTRHBuilder      = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing          = cms.int32(6),
        maxRing          = cms.int32(7)
    )
)

# Triplet TrackingRegion
from RecoTracker.FinalTrackSelectors.displacedRegionalStepInputTracks_cfi import displacedRegionalStepInputTracks
from RecoVertex.V0Producer.generalV0Candidates_cfi import generalV0Candidates as _generalV0Candidates
from RecoTracker.DisplacedRegionalTracking.displacedRegionProducer_cfi import displacedRegionProducer as _displacedRegionProducer
displacedRegionalStepSeedingV0Candidates = _generalV0Candidates.clone(
    trackRecoAlgorithm = "displacedRegionalStepInputTracks",
    doLambdas = False,
    doFit = False,
    useRefTracks = False,
    vtxDecayXYCut = 1.,
    ssVtxDecayXYCut = 5.,
    allowSS = True,
    innerTkDCACut = 0.2,
    allowWideAngleVtx = True,
    mPiPiCut = 13000.,
    cosThetaXYCut = 0.,
    kShortMassCut = 13000.,
)
displacedRegionalStepSeedingVertices = _displacedRegionProducer.clone(
    minRadius = 2.0,
    discriminatorCut = 0.5,
    trackClusters = ["displacedRegionalStepSeedingV0Candidates", "Kshort"],
)

from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
displacedRegionalStepTrackingRegionsTripl = _globalTrackingRegionWithVertices.clone(RegionPSet = dict(
    originRadius = 1.0,
    fixedError = 1.0,
    VertexCollection = "displacedRegionalStepSeedingVertices",
    useFakeVertices = True,
    ptMin = 0.55,
    allowEmpty = True
))

# Triplet seeding
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import ClusterShapeHitFilterESProducer as _ClusterShapeHitFilterESProducer
displacedRegionalStepClusterShapeHitFilter = _ClusterShapeHitFilterESProducer.clone(
    ComponentName = 'displacedRegionalStepClusterShapeHitFilter',
    doStripShapeCut = False,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight')
)

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
displacedRegionalStepHitDoubletsTripl = _hitPairEDProducer.clone(
    seedingLayers = "displacedRegionalStepSeedLayersTripl",
    trackingRegions = "displacedRegionalStepTrackingRegionsTripl",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoTracker.TkSeedGenerator.multiHitFromChi2EDProducer_cfi import multiHitFromChi2EDProducer as _multiHitFromChi2EDProducer
displacedRegionalStepHitTripletsTripl = _multiHitFromChi2EDProducer.clone(
    doublets = "displacedRegionalStepHitDoubletsTripl",
    extraPhiKDBox = 0.01,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
_displacedRegionalStepSeedComparitorPSet = dict(
    ComponentName = 'CombinedSeedComparitor',
    mode = cms.string("and"),
    comparitors = cms.VPSet(
        cms.PSet(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
            ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
            FilterAtHelixStage = cms.bool(True),
            FilterPixelHits = cms.bool(False),
            FilterStripHits = cms.bool(True),
            ClusterShapeHitFilterName = cms.string('displacedRegionalStepClusterShapeHitFilter'),
            ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache") # not really needed here since FilterPixelHits=False
        )
    )
)

displacedRegionalStepSeedsTripl = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(#empirically better than 'SeedFromConsecutiveHitsTripletOnlyCreator'
    seedingHitSets = "displacedRegionalStepHitTripletsTripl",
    SeedComparitorPSet = _displacedRegionalStepSeedComparitorPSet,
)

from RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi import StripSubClusterShapeSeedFilter as _StripSubClusterShapeSeedFilter
from Configuration.ProcessModifiers.approxSiStripClusters_cff import approxSiStripClusters
(~approxSiStripClusters).toModify(displacedRegionalStepSeedsTripl.SeedComparitorPSet.comparitors, func = lambda list: list.append(_StripSubClusterShapeSeedFilter.clone()) )

# PAIR SEEDING LAYERS
displacedRegionalStepSeedLayersPair = _mod.seedingLayersEDProducer.clone(
    layerList = ['TOB1+TEC1_pos','TOB1+TEC1_neg',
                 'TEC1_pos+TEC2_pos','TEC1_neg+TEC2_neg',
                 'TEC2_pos+TEC3_pos','TEC2_neg+TEC3_neg',
                 'TEC3_pos+TEC4_pos','TEC3_neg+TEC4_neg',
                 'TEC4_pos+TEC5_pos','TEC4_neg+TEC5_neg',
                 'TEC5_pos+TEC6_pos','TEC5_neg+TEC6_neg',
                 'TEC6_pos+TEC7_pos','TEC6_neg+TEC7_neg'],
    TOB = dict(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
         skipClusters   = cms.InputTag('displacedRegionalStepClusters')
    ),
    TEC = dict(
        matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
        skipClusters = cms.InputTag('displacedRegionalStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)
# Pair TrackingRegion
displacedRegionalStepTrackingRegionsPair = _globalTrackingRegionWithVertices.clone(RegionPSet = dict(
    originRadius = 1.0,
    fixedError = 1.0,
    VertexCollection = "displacedRegionalStepSeedingVertices",
    useFakeVertices = True,
    ptMin = 0.6,
    allowEmpty = True
))

# Pair seeds
displacedRegionalStepHitDoubletsPair = _hitPairEDProducer.clone(
    seedingLayers = "displacedRegionalStepSeedLayersPair",
    trackingRegions = "displacedRegionalStepTrackingRegionsPair",
    produceSeedingHitSets = True,
    maxElementTotal = 12000000,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
displacedRegionalStepSeedsPair = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "displacedRegionalStepHitDoubletsPair",
    SeedComparitorPSet = _displacedRegionalStepSeedComparitorPSet,
)

# Combined seeds
import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
displacedRegionalStepSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone(
    seedCollections = ['displacedRegionalStepSeedsTripl', 'displacedRegionalStepSeedsPair']
)

# QUALITY CUTS DURING TRACK BUILDING (for inwardss and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_displacedRegionalStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 5,
    minPt = 0.1,
    minHitsMinPt = 3
    )
displacedRegionalStepTrajectoryFilter = _displacedRegionalStepTrajectoryFilterBase.clone(
    seedPairPenalty = 1,
)

displacedRegionalStepInOutTrajectoryFilter = displacedRegionalStepTrajectoryFilter.clone(
    minimumNumberOfHits = 4,
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
displacedRegionalStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'displacedRegionalStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 16.0,
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
displacedRegionalStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter       = dict(refToPSet_ = 'displacedRegionalStepTrajectoryFilter'),
    inOutTrajectoryFilter  = dict(refToPSet_ = 'displacedRegionalStepInOutTrajectoryFilter'),
    useSameTrajFilter      = False,
    minNrOfHitsForRebuild  = 4,
    alwaysUseInvalidHits   = False,
    maxCand                = 2,
    estimator              = 'displacedRegionalStepChi2Est',
    #startSeedHitsInRebuild = True,
    maxDPhiForLooperReconstruction = 2.0,
    maxPtForLooperReconstruction   = 0.7,
)
trackingNoLoopers.toModify(displacedRegionalStepTrajectoryBuilder,
                           maxPtForLooperReconstruction = 0.0)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
# Give handle for CKF for HI
_displacedRegionalStepTrackCandidatesCkf = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'displacedRegionalStepSeeds',
    clustersToSkip              = 'displacedRegionalStepClusters',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner       = 50,
    onlyPixelHitsForSeedCleaner = False,
    TrajectoryBuilderPSet       = dict(refToPSet_ = 'displacedRegionalStepTrajectoryBuilder'),
    doSeedingRegionRebuilding   = True,
    useHitsSplitting            = True,
    cleanTrajectoryAfterInOut   = True,
    TrajectoryCleaner = 'displacedRegionalStepTrajectoryCleanerBySharedHits',
)
displacedRegionalStepTrackCandidates = _displacedRegionalStepTrackCandidatesCkf.clone()

from Configuration.ProcessModifiers.trackingMkFitDisplacedRegionalStep_cff import trackingMkFitDisplacedRegionalStep
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
import RecoTracker.MkFit.mkFitIterationConfigESProducer_cfi as mkFitIterationConfigESProducer_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
displacedRegionalStepTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
    seeds = 'displacedRegionalStepSeeds',
)
displacedRegionalStepTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
    ComponentName = 'displacedRegionalStepTrackCandidatesMkFitConfig',
    config = 'RecoTracker/MkFit/data/mkfit-phase1-tobTecStep.json',
)
displacedRegionalStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
    seeds = 'displacedRegionalStepTrackCandidatesMkFitSeeds',
    config = ('', 'displacedRegionalStepTrackCandidatesMkFitConfig'),
    clustersToSkip = 'displacedRegionalStepClusters',
)
trackingMkFitDisplacedRegionalStep.toReplaceWith(displacedRegionalStepTrackCandidates, mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
    seeds = 'displacedRegionalStepSeeds',
    mkFitSeeds = 'displacedRegionalStepTrackCandidatesMkFitSeeds',
    tracks = 'displacedRegionalStepTrackCandidatesMkFit',
))

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
displacedRegionalStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = 'displacedRegionalStepTrajectoryCleanerBySharedHits',
    fractionShared = 0.09,
    allowSharedFirstHit = True
    )

# TRACK FITTING AND SMOOTHING OPTIONS
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
displacedRegionalStepFitterSmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'displacedRegionalStepFitterSmoother',
    EstimateCut = 30,
    MinNumberOfHits = 7,
    Fitter = 'displacedRegionalStepRKFitter',
    Smoother = 'displacedRegionalStepRKSmoother'
    )

displacedRegionalStepFitterSmootherForLoopers = displacedRegionalStepFitterSmoother.clone(
    ComponentName = 'displacedRegionalStepFitterSmootherForLoopers',
    Fitter = 'displacedRegionalStepRKFitterForLoopers',
    Smoother = 'displacedRegionalStepRKSmootherForLoopers'
)

# Also necessary to specify minimum number of hits after final track fit
displacedRegionalStepRKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = 'displacedRegionalStepRKFitter',
    minHits = 7
)

displacedRegionalStepRKTrajectoryFitterForLoopers = displacedRegionalStepRKTrajectoryFitter.clone(
    ComponentName = 'displacedRegionalStepRKFitterForLoopers',
    Propagator = 'PropagatorWithMaterialForLoopers',
)

displacedRegionalStepRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = 'displacedRegionalStepRKSmoother',
    errorRescaling = 10.0,
    minHits = 7
)

displacedRegionalStepRKTrajectorySmootherForLoopers = displacedRegionalStepRKTrajectorySmoother.clone(
    ComponentName = 'displacedRegionalStepRKSmootherForLoopers',
    Propagator = 'PropagatorWithMaterialForLoopers',
)

import TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi
displacedRegionalFlexibleKFFittingSmoother = TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi.FlexibleKFFittingSmoother.clone(
    ComponentName = 'displacedRegionalFlexibleKFFittingSmoother',
    standardFitter = 'displacedRegionalStepFitterSmoother',
    looperFitter = 'displacedRegionalStepFitterSmootherForLoopers',
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi
displacedRegionalStepTracks = RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi.TrackProducer.clone(
    src           = 'displacedRegionalStepTrackCandidates',
    AlgorithmName = 'displacedRegionalStep',
    #Fitter = 'displacedRegionalStepFitterSmoother',
    Fitter        = 'displacedRegionalFlexibleKFFittingSmoother',
)

# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
displacedRegionalStepClassifier1 = TrackMVAClassifierDetached.clone(
    src = 'displacedRegionalStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorIter6_13TeV'),
    qualityCuts = [-0.6,-0.45,-0.3]
)

displacedRegionalStepClassifier2 = TrackMVAClassifierPrompt.clone(
    src = 'displacedRegionalStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorIter0_13TeV'),
    qualityCuts = [0.0,0.0,0.0]
)

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
displacedRegionalStep = ClassifierMerger.clone(
    inputClassifiers=['displacedRegionalStepClassifier1','displacedRegionalStepClassifier2']
)

from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toReplaceWith(displacedRegionalStep, displacedRegionalStepClassifier1.clone(
     mva = dict(GBRForestLabel = 'MVASelectorTobTecStep_Phase1'),
     qualityCuts = [-0.6,-0.45,-0.3]
))

from RecoTracker.FinalTrackSelectors.trackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_CKF_cfi import *
trackdnn.toReplaceWith(displacedRegionalStep, trackTfClassifier.clone(
     src         = 'displacedRegionalStepTracks',
     qualityCuts = qualityCutDictionary.DisplacedRegionalStep.value()
))

DisplacedRegionalStepTask = cms.Task(displacedRegionalStepClusters,
                          displacedRegionalStepSeedLayersTripl,
                          displacedRegionalStepInputTracks,
                          displacedRegionalStepSeedingV0Candidates,
                          displacedRegionalStepSeedingVertices,
                          displacedRegionalStepTrackingRegionsTripl,
                          displacedRegionalStepHitDoubletsTripl,
                          displacedRegionalStepHitTripletsTripl,
                          displacedRegionalStepSeedsTripl,
                          displacedRegionalStepSeedLayersPair,
                          displacedRegionalStepTrackingRegionsPair,
                          displacedRegionalStepHitDoubletsPair,
                          displacedRegionalStepSeedsPair,
                          displacedRegionalStepSeeds,
                          displacedRegionalStepTrackCandidates,
                          displacedRegionalStepTracks,
                          displacedRegionalStepClassifier1,displacedRegionalStepClassifier2,
                          displacedRegionalStep)
DisplacedRegionalStep = cms.Sequence(DisplacedRegionalStepTask)

_DisplacedRegionalStepTask_trackingMkFit = DisplacedRegionalStepTask.copy()
_DisplacedRegionalStepTask_trackingMkFit.add(displacedRegionalStepTrackCandidatesMkFitSeeds, displacedRegionalStepTrackCandidatesMkFit, displacedRegionalStepTrackCandidatesMkFitConfig)
trackingMkFitDisplacedRegionalStep.toReplaceWith(DisplacedRegionalStepTask, _DisplacedRegionalStepTask_trackingMkFit)
