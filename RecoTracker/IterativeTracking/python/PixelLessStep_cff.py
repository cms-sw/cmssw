import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

from Configuration.Eras.Modifier_fastSim_cff import fastSim

# for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

# for no-loopers
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

##########################################################################
# Large impact parameter tracking using TIB/TID/TEC stereo layer seeding #
##########################################################################

pixelLessStepClusters = _cfg.clusterRemoverForIter('PixelLessStep')
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(pixelLessStepClusters, _cfg.clusterRemoverForIter('PixelLessStep', _eraName, _postfix))



# SEEDING LAYERS
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *
import RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi as _mod

pixelLessStepSeedLayers = _mod.seedingLayersEDProducer.clone(
    layerList = [
    #TIB
    'TIB1+TIB2+MTIB3','TIB1+TIB2+MTIB4',
    #TIB+TID
    'TIB1+TIB2+MTID1_pos','TIB1+TIB2+MTID1_neg',
    #TID
    'TID1_pos+TID2_pos+TID3_pos','TID1_neg+TID2_neg+TID3_neg',#ring 1-2 (matched)
    'TID1_pos+TID2_pos+MTID3_pos','TID1_neg+TID2_neg+MTID3_neg',#ring 3 (mono)
    'TID1_pos+TID2_pos+MTEC1_pos','TID1_neg+TID2_neg+MTEC1_neg',
    #TID+TEC RING 1-3
    'TID2_pos+TID3_pos+TEC1_pos','TID2_neg+TID3_neg+TEC1_neg',#ring 1-2 (matched)
    'TID2_pos+TID3_pos+MTEC1_pos','TID2_neg+TID3_neg+MTEC1_neg',#ring 3 (mono)
    #TEC RING 1-3
    'TEC1_pos+TEC2_pos+TEC3_pos', 'TEC1_neg+TEC2_neg+TEC3_neg',
    'TEC1_pos+TEC2_pos+MTEC3_pos','TEC1_neg+TEC2_neg+MTEC3_neg',
    'TEC1_pos+TEC2_pos+TEC4_pos', 'TEC1_neg+TEC2_neg+TEC4_neg',
    'TEC1_pos+TEC2_pos+MTEC4_pos','TEC1_neg+TEC2_neg+MTEC4_neg',
    'TEC2_pos+TEC3_pos+TEC4_pos', 'TEC2_neg+TEC3_neg+TEC4_neg',
    'TEC2_pos+TEC3_pos+MTEC4_pos','TEC2_neg+TEC3_neg+MTEC4_neg',
    'TEC2_pos+TEC3_pos+TEC5_pos', 'TEC2_neg+TEC3_neg+TEC5_neg',
    'TEC2_pos+TEC3_pos+TEC6_pos', 'TEC2_neg+TEC3_neg+TEC6_neg',
    'TEC3_pos+TEC4_pos+TEC5_pos', 'TEC3_neg+TEC4_neg+TEC5_neg',
    'TEC3_pos+TEC4_pos+MTEC5_pos','TEC3_neg+TEC4_neg+MTEC5_neg',
    'TEC3_pos+TEC5_pos+TEC6_pos', 'TEC3_neg+TEC5_neg+TEC6_neg',
    'TEC4_pos+TEC5_pos+TEC6_pos', 'TEC4_neg+TEC5_neg+TEC6_neg'    
    ],
    TIB = dict(
         TTRHBuilder    = cms.string('WithTrackAngle'), 
         clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
         skipClusters   = cms.InputTag('pixelLessStepClusters')
    ),
    MTIB = dict(
         TTRHBuilder    = cms.string('WithTrackAngle'), 
         clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         skipClusters   = cms.InputTag('pixelLessStepClusters'),
         rphiRecHits    = cms.InputTag('siStripMatchedRecHits','rphiRecHit')
    ),
    TID = dict(
        matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), 
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    MTID = dict(
        rphiRecHits    = cms.InputTag('siStripMatchedRecHits','rphiRecHit'),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), 
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(3),
        maxRing = cms.int32(3)
    ),
    TEC = dict(
        matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), 
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    MTEC = dict(
        rphiRecHits = cms.InputTag('siStripMatchedRecHits','rphiRecHit'),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), 
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(3),
        maxRing = cms.int32(3)
    )
)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(pixelLessStepSeedLayers,
    layerList = [
        'TIB1+TIB2',
        'TID1_pos+TID2_pos','TID2_pos+TID3_pos',
        'TEC1_pos+TEC2_pos','TEC2_pos+TEC3_pos','TEC3_pos+TEC4_pos','TEC3_pos+TEC5_pos','TEC4_pos+TEC5_pos',
        'TID1_neg+TID2_neg','TID2_neg+TID3_neg',
        'TEC1_neg+TEC2_neg','TEC2_neg+TEC3_neg','TEC3_neg+TEC4_neg','TEC3_neg+TEC5_neg','TEC4_neg+TEC5_neg'
    ],
    TIB = dict(clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')),
    TID = dict(clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')),
    TEC = dict(clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')),
    MTIB = None,
    MTID = None,
    MTEC = None,
)
from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
vectorHits.toModify(pixelLessStepSeedLayers,
    layerList = [
        'TOB1+TOB2', 'TOB2+TOB3',
#        'TOB3+TOB4', 'TOB4+TOB5', 
        'TID1_pos+TID2_pos', 'TID1_neg+TID2_neg'
    ],
    TOB = dict(
         TTRHBuilder      = cms.string('WithTrackAngle'), 
         clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
         vectorRecHits    = cms.InputTag("siPhase2VectorHits", 'vectorHitsAccepted'),
         skipClusters     = cms.InputTag('pixelLessStepClusters')
    ),
    TIB = None,
    TID = dict(
         clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
         vectorRecHits    = cms.InputTag("siPhase2VectorHits", 'accepted'),
         maxRing          = 8
    ),
    TEC  = None,
    MTIB = None,
    MTID = None,
    MTEC = None,
)
# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpotFixedZ_cfi import globalTrackingRegionFromBeamSpotFixedZ as _globalTrackingRegionFromBeamSpotFixedZ
pixelLessStepTrackingRegions = _globalTrackingRegionFromBeamSpotFixedZ.clone(
    RegionPSet = dict(
        ptMin            = 0.4,
        originHalfLength = 12.0,
        originRadius     = 1.0)
)
trackingLowPU.toModify(pixelLessStepTrackingRegions, RegionPSet = dict(
    ptMin            = 0.7,
    originHalfLength = 10.0,
    originRadius     = 2.0,
))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from RecoTracker.IterativeTracking.MixedTripletStep_cff import _mixedTripletStepTrackingRegionsCommon_pp_on_HI
(pp_on_XeXe_2017 | pp_on_AA).toReplaceWith(pixelLessStepTrackingRegions, 
                _mixedTripletStepTrackingRegionsCommon_pp_on_HI.clone(RegionPSet=dict(
                    ptMinScaling4BigEvts = False,
                    fixedError           = 3.0,
                    ptMin                = 2.0,
                    originRadius         = 1.0 )
                )
)


# seeding
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import ClusterShapeHitFilterESProducer as _ClusterShapeHitFilterESProducer
pixelLessStepClusterShapeHitFilter = _ClusterShapeHitFilterESProducer.clone(
    ComponentName    = 'pixelLessStepClusterShapeHitFilter',
    doStripShapeCut  = cms.bool(False),
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight')
)

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
pixelLessStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers   = 'pixelLessStepSeedLayers',
    trackingRegions = 'pixelLessStepTrackingRegions',
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoTracker.TkSeedGenerator.multiHitFromChi2EDProducer_cfi import multiHitFromChi2EDProducer as _multiHitFromChi2EDProducer
pixelLessStepHitTriplets = _multiHitFromChi2EDProducer.clone(
    doublets = 'pixelLessStepHitDoublets',
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
from RecoTracker.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi import StripSubClusterShapeSeedFilter as _StripSubClusterShapeSeedFilter
pixelLessStepSeeds = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets = 'pixelLessStepHitTriplets',
    SeedComparitorPSet = dict(
        ComponentName = 'CombinedSeedComparitor',
        mode = cms.string('and'),
        comparitors = cms.VPSet(
            cms.PSet(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
                ComponentName      = cms.string('PixelClusterShapeSeedComparitor'),
                FilterAtHelixStage = cms.bool(True),
                FilterPixelHits    = cms.bool(False),
                FilterStripHits    = cms.bool(True),
                ClusterShapeHitFilterName = cms.string('pixelLessStepClusterShapeHitFilter'),
                ClusterShapeCacheSrc      = cms.InputTag('siPixelClusterShapeCache') # not really needed here since FilterPixelHits=False
            ),
            _StripSubClusterShapeSeedFilter.clone()
        )
    )
)

trackingLowPU.toModify(pixelLessStepHitDoublets, produceSeedingHitSets=True, produceIntermediateHitDoublets=False)
trackingLowPU.toModify(pixelLessStepSeeds,
    seedingHitSets = 'pixelLessStepHitDoublets',
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName      = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits    = cms.bool(False),
        FilterStripHits    = cms.bool(True),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc      = cms.InputTag('siPixelClusterShapeCache') # not really needed here since FilterPixelHits=False
    )
)
#fastsim
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet
_fastSim_pixelLessStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = 'pixelLessStepTrackingRegions',
    hitMasks        = cms.InputTag('pixelLessStepMasks'),
    seedFinderSelector = dict( MultiHitGeneratorFactory = _hitSetProducerToFactoryPSet(pixelLessStepHitTriplets).clone(
                              refitHits = False),
                              layerList = pixelLessStepSeedLayers.layerList.value()
))
fastSim.toReplaceWith(pixelLessStepSeeds,_fastSim_pixelLessStepSeeds)

vectorHits.toModify(pixelLessStepHitDoublets, produceSeedingHitSets=True, produceIntermediateHitDoublets=False)
vectorHits.toModify(pixelLessStepSeeds, 
    seedingHitSets = "pixelLessStepHitDoublets",
    SeedComparitorPSet = dict(
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        FilterAtHelixStage = cms.bool(False),
        FilterStripHits = cms.bool(False),
    )
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_pixelLessStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits         = 0,
    minimumNumberOfHits = 4,
    minPt               = 0.1
)
pixelLessStepTrajectoryFilter = _pixelLessStepTrajectoryFilterBase.clone(
    seedPairPenalty = 1,
)
trackingLowPU.toReplaceWith(pixelLessStepTrajectoryFilter, _pixelLessStepTrajectoryFilterBase)
(pp_on_XeXe_2017 | pp_on_AA).toModify(pixelLessStepTrajectoryFilter, minPt=2.0)

vectorHits.toReplaceWith(pixelLessStepTrajectoryFilter, _pixelLessStepTrajectoryFilterBase)
vectorHits.toModify(pixelLessStepTrajectoryFilter, minimumNumberOfHits = 4, maxLostHits = 1)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
pixelLessStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName    = 'pixelLessStepChi2Est',
    nSigma           = 3.0,
    MaxChi2          = 16.0,
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))
)
trackingLowPU.toModify(pixelLessStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')
)

vectorHits.toModify(pixelLessStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
    MaxChi2 = 30.0
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
pixelLessStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilderIterativeDefault.clone(
    trajectoryFilter       = dict(refToPSet_ = 'pixelLessStepTrajectoryFilter'),
    minNrOfHitsForRebuild  = 4,
    maxCand                = 2,
    alwaysUseInvalidHits   = False,
    estimator              = 'pixelLessStepChi2Est',
    maxDPhiForLooperReconstruction = 2.0,
    maxPtForLooperReconstruction   = 0.7,
)
trackingNoLoopers.toModify(pixelLessStepTrajectoryBuilder,
                           maxPtForLooperReconstruction = 0.0)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
# Give handle for CKF for HI
_pixelLessStepTrackCandidatesCkf = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidatesIterativeDefault.clone(
    src                   = 'pixelLessStepSeeds',
    clustersToSkip        = 'pixelLessStepClusters',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    #onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'pixelLessStepTrajectoryBuilder'),
    TrajectoryCleaner     = 'pixelLessStepTrajectoryCleanerBySharedHits',
)
pixelLessStepTrackCandidates = _pixelLessStepTrackCandidatesCkf.clone()

from Configuration.ProcessModifiers.trackingMkFitPixelLessStep_cff import trackingMkFitPixelLessStep
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
import RecoTracker.MkFit.mkFitIterationConfigESProducer_cfi as mkFitIterationConfigESProducer_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
pixelLessStepTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
    seeds = 'pixelLessStepSeeds',
)
pixelLessStepTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
    ComponentName = 'pixelLessStepTrackCandidatesMkFitConfig',
    config = 'RecoTracker/MkFit/data/mkfit-phase1-pixelLessStep.json',
)
pixelLessStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
    seeds = 'pixelLessStepTrackCandidatesMkFitSeeds',
    config = ('', 'pixelLessStepTrackCandidatesMkFitConfig'),
    clustersToSkip = 'pixelLessStepClusters',
)
trackingMkFitPixelLessStep.toReplaceWith(pixelLessStepTrackCandidates, mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
    seeds = 'pixelLessStepSeeds',
    mkFitSeeds = 'pixelLessStepTrackCandidatesMkFitSeeds',
    tracks = 'pixelLessStepTrackCandidatesMkFit',
    candMVASel = False,
    candWP = -0.7,
))
(pp_on_XeXe_2017 | pp_on_AA).toModify(pixelLessStepTrackCandidatesMkFitConfig, minPt=2.0)

import FastSimulation.Tracking.TrackCandidateProducer_cfi
fastSim.toReplaceWith(pixelLessStepTrackCandidates,
                      FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
        src = 'pixelLessStepSeeds',
        MinNumberOfCrossedLayers = 6, # ?
        hitMasks = cms.InputTag('pixelLessStepMasks')
        )
)

vectorHits.toModify(pixelLessStepTrackCandidates,
    phase2clustersToSkip = 'pixelLessStepClusters',
    clustersToSkip = ''
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
pixelLessStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName       = 'pixelLessStepTrajectoryCleanerBySharedHits',
    fractionShared      = 0.11,
    allowSharedFirstHit = True
)
trackingLowPU.toModify(pixelLessStepTrajectoryCleanerBySharedHits, fractionShared = 0.19)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi
pixelLessStepTracks = RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi.TrackProducerIterativeDefault.clone(
    src           = 'pixelLessStepTrackCandidates',
    AlgorithmName = 'pixelLessStep',
    Fitter        = 'FlexibleKFFittingSmoother'
)
fastSim.toModify(pixelLessStepTracks, TTRHBuilder = 'WithoutRefit')

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify(pixelLessStepTracks, TrajectoryInEvent = True)

# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
pixelLessStepClassifier1 = TrackMVAClassifierPrompt.clone(
    src         = 'pixelLessStepTracks',
    mva         = dict(GBRForestLabel = 'MVASelectorIter5_13TeV'),
    qualityCuts = [-0.4,0.0,0.4]
)
fastSim.toModify(pixelLessStepClassifier1, vertices = 'firstStepPrimaryVerticesBeforeMixing' )

pixelLessStepClassifier2 = TrackMVAClassifierPrompt.clone(
    src         = 'pixelLessStepTracks',
    mva         = dict(GBRForestLabel = 'MVASelectorIter0_13TeV'),
    qualityCuts = [-0.0,0.0,0.0]
)
fastSim.toModify(pixelLessStepClassifier2, vertices = 'firstStepPrimaryVerticesBeforeMixing' )

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
pixelLessStep = ClassifierMerger.clone(
    inputClassifiers=['pixelLessStepClassifier1','pixelLessStepClassifier2']
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1

trackingPhase1.toReplaceWith(pixelLessStep, pixelLessStepClassifier1.clone(
    mva         = dict(GBRForestLabel = 'MVASelectorPixelLessStep_Phase1'),
    qualityCuts = [-0.4,0.0,0.4]
))

from RecoTracker.FinalTrackSelectors.trackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_CKF_cfi import *
trackdnn.toReplaceWith(pixelLessStep, trackTfClassifier.clone(
    mva         = dict(tfDnnLabel  = 'trackSelectionTfPLess'),
    src         = 'pixelLessStepTracks',
    qualityCuts = qualityCutDictionary.PixelLessStep.value()
))
(trackdnn & fastSim).toModify(pixelLessStep,vertices = 'firstStepPrimaryVerticesBeforeMixing')

((~trackingMkFitPixelLessStep) & trackdnn).toModify(pixelLessStep, mva = dict(tfDnnLabel  = 'trackSelectionTf_CKF'),
                                                    qualityCuts = [-0.82, -0.61, -0.16])

pp_on_AA.toModify(pixelLessStep, qualityCuts = [-0.4,0.0,0.8])

# For LowPU
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
pixelLessStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='pixelLessStepTracks',
    useAnyMVA      = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter5'),
    trackSelectors = cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixelLessStepLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.3, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'pixelLessStepTight',
            preFilterName = 'pixelLessStepLoose',
            chi2n_par = 0.35,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'pixelLessStepTight',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
        ),
    ),
    vertices = 'pixelVertices'#end of vpset
) #end of clone

vectorHits.toModify(pixelLessStepSelector, 
    GBRForestLabel = None,
    useAnyMVA = None,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixelLessStepLoose',
            chi2n_par = 1.0,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 0,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 0,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 1.0, 4.0 ),
            dz_par2 = ( 1.0, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'pixelLessStepTight',
            preFilterName = 'pixelLessStepLoose',
            chi2n_par = 0.35,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'pixelLessStepTight',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 1,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 0,
            d0_par1 = ( 100., 4.0 ),
            dz_par1 = ( 100., 4.0 ),
            d0_par2 = ( 100., 4.0 ),
            dz_par2 = ( 100., 4.0 )
        ),
    ),
    vertices = 'firstStepPrimaryVertices'
) 

vectorHits.toModify(pixelLessStepSelector.trackSelectors[2], name = 'pixelLessStep')


PixelLessStepTask = cms.Task(pixelLessStepClusters,
                             pixelLessStepSeedLayers,
                             pixelLessStepTrackingRegions,
                             pixelLessStepHitDoublets,
                             pixelLessStepHitTriplets,
                             pixelLessStepSeeds,
                             pixelLessStepTrackCandidates,
                             pixelLessStepTracks,
                             pixelLessStepClassifier1,pixelLessStepClassifier2,
                             pixelLessStep)
PixelLessStep = cms.Sequence(PixelLessStepTask)

_PixelLessStepTask_trackingMkFit = PixelLessStepTask.copy()
_PixelLessStepTask_trackingMkFit.add(pixelLessStepTrackCandidatesMkFitSeeds, pixelLessStepTrackCandidatesMkFit, pixelLessStepTrackCandidatesMkFit)
trackingMkFitPixelLessStep.toReplaceWith(PixelLessStepTask, _PixelLessStepTask_trackingMkFit)

_PixelLessStepTask_LowPU = PixelLessStepTask.copyAndExclude([pixelLessStepHitTriplets, pixelLessStepClassifier1, pixelLessStepClassifier2])
_PixelLessStepTask_LowPU.replace(pixelLessStep, pixelLessStepSelector)
trackingLowPU.toReplaceWith(PixelLessStepTask, _PixelLessStepTask_LowPU)
#fastsim
from FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi import maskProducerFromClusterRemover
pixelLessStepMasks = maskProducerFromClusterRemover(pixelLessStepClusters)
fastSim.toReplaceWith(PixelLessStepTask,
                      cms.Task(pixelLessStepMasks
                                   ,pixelLessStepTrackingRegions
                                   ,pixelLessStepSeeds
                                   ,pixelLessStepTrackCandidates
                                   ,pixelLessStepTracks
                                   ,pixelLessStepClassifier1,pixelLessStepClassifier2
                                   ,pixelLessStep                             
                                   )
)
from RecoLocalTracker.SiPhase2VectorHitBuilder.siPhase2VectorHits_cfi import *
from RecoTracker.TkSeedGenerator.SeedingOTEDProducer_cfi import SeedingOTEDProducer as _SeedingOTEDProducer
pixelLessStepSeeds_vectorHits = _SeedingOTEDProducer.clone()

_PixelLessStepTask_vectorHits = cms.Task(siPhase2VectorHits,
			     pixelLessStepClusters,
                             pixelLessStepSeeds,
                             pixelLessStepTrackCandidates,
                             pixelLessStepTracks,
                             pixelLessStepSelector)
_PixelLessStep_vectorHits = cms.Sequence(_PixelLessStepTask_vectorHits)
vectorHits.toReplaceWith(pixelLessStepSeeds,pixelLessStepSeeds_vectorHits)
vectorHits.toReplaceWith(PixelLessStepTask, _PixelLessStepTask_vectorHits)
vectorHits.toReplaceWith(PixelLessStep, _PixelLessStep_vectorHits)
