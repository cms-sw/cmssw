import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
from Configuration.Eras.Modifier_fastSim_cff import fastSim

# for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

# for no-loopers
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

# NEW CLUSTERS (remove previously used clusters)
lowPtQuadStepClusters = _cfg.clusterRemoverForIter('LowPtQuadStep')
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(lowPtQuadStepClusters, _cfg.clusterRemoverForIter('LowPtQuadStep', _eraName, _postfix))


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi
lowPtQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi.PixelLayerQuadruplets.clone(
    BPix = dict(skipClusters = cms.InputTag('lowPtQuadStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('lowPtQuadStepClusters'))
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
lowPtQuadStepTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin        = 0.15,
    originRadius = 0.02,
    nSigmaZ      = 4.0 )
)
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(lowPtQuadStepTrackingRegions, RegionPSet = dict(ptMin = 0.35,originRadius = 0.025))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
(pp_on_XeXe_2017 | pp_on_AA).toReplaceWith(lowPtQuadStepTrackingRegions, 
                _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
                    fixedError   = 0.5,
                    ptMin        = 0.49,
                    originRadius = 0.02 )
                )
)
from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(lowPtQuadStepTrackingRegions,RegionPSet = dict(
     ptMin        = 0.05,
     originRadius = 0.2, )
)

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
lowPtQuadStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers   = 'lowPtQuadStepSeedLayers',
    trackingRegions = 'lowPtQuadStepTrackingRegions',
    layerPairs      = [0,1,2], # layer pairs (0,1), (1,2), (2,3)
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoTracker.PixelSeeding.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtQuadStepHitQuadruplets = _caHitQuadrupletEDProducer.clone(
    doublets = 'lowPtQuadStepHitDoublets',
    extraHitRPhitolerance = _pixelTripletHLTEDProducer.extraHitRPhitolerance,
    SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 1000, value2 = 150,
    ),
    useBendingCorrection = True,
    fitFastCircle        = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut           = 0.0017,
    CAPhiCut             = 0.3,
)
trackingPhase2PU140.toModify(lowPtQuadStepHitQuadruplets,CAThetaCut = 0.0015,CAPhiCut = 0.25)
highBetaStar_2018.toModify(lowPtQuadStepHitQuadruplets,CAThetaCut = 0.0034,CAPhiCut = 0.6)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
lowPtQuadStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = 'lowPtQuadStepHitQuadruplets',
)

#For FastSim phase1 tracking
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet
_fastSim_lowPtQuadStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = 'lowPtQuadStepTrackingRegions',
    hitMasks        = cms.InputTag('lowPtQuadStepMasks'),
    seedFinderSelector = dict( CAHitQuadrupletGeneratorFactory = _hitSetProducerToFactoryPSet(lowPtQuadStepHitQuadruplets).clone(
            SeedComparitorPSet = dict(ComponentName = 'none')),
                               layerList = lowPtQuadStepSeedLayers.layerList.value(),
                               #new parameters required for phase1 seeding
                               BPix      = dict(TTRHBuilder = 'WithoutRefit', HitProducer = 'TrackingRecHitProducer',),
                               FPix      = dict(TTRHBuilder = 'WithoutRefit', HitProducer = 'TrackingRecHitProducer',),
                               layerPairs = lowPtQuadStepHitDoublets.layerPairs.value()
                               )
)
fastSim.toReplaceWith(lowPtQuadStepSeeds,_fastSim_lowPtQuadStepSeeds)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_lowPtQuadStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt               = 0.075,
)
lowPtQuadStepTrajectoryFilterBase = _lowPtQuadStepTrajectoryFilterBase.clone(
    maxCCCLostHits     = 0,
    minGoodStripCharge = dict(refToPSet_ = 'SiStripClusterChargeCutLoose')
)
trackingPhase2PU140.toReplaceWith(lowPtQuadStepTrajectoryFilterBase, _lowPtQuadStepTrajectoryFilterBase)

(pp_on_XeXe_2017 | pp_on_AA).toModify(lowPtQuadStepTrajectoryFilterBase, minPt=0.49)

from RecoTracker.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtQuadStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('lowPtQuadStepTrajectoryFilterBase'))]
)
trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryFilter,
    filters = lowPtQuadStepTrajectoryFilter.filters.value() + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName    = 'lowPtQuadStepChi2Est',
    nSigma           = 3.0,
    MaxChi2          = 9.0,
    clusterChargeCut = dict(refToPSet_ = ('SiStripClusterChargeCutTight')),
)
trackingPhase2PU140.toModify(lowPtQuadStepChi2Est,
    MaxChi2          = 16.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilderIterativeDefault.clone(
    trajectoryFilter       = dict(refToPSet_ = 'lowPtQuadStepTrajectoryFilter'),
    maxCand                = 4,
    estimator              = 'lowPtQuadStepChi2Est',
    maxDPhiForLooperReconstruction = 2.0,
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = 0.7,
)
trackingNoLoopers.toModify(lowPtQuadStepTrajectoryBuilder,
                           maxPtForLooperReconstruction = 0.0)
trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryBuilder,
    minNrOfHitsForRebuild      = 1,
    keepOriginalIfRebuildFails = True,
)


# MAKING OF TRACK CANDIDATES
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits as _trajectoryCleanerBySharedHits
lowPtQuadStepTrajectoryCleanerBySharedHits = _trajectoryCleanerBySharedHits.clone(
    ComponentName       = 'lowPtQuadStepTrajectoryCleanerBySharedHits',
    fractionShared      = 0.16,
    allowSharedFirstHit = True
)
trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidatesIterativeDefault.clone(
    src = 'lowPtQuadStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner       = 50,
    onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet       = dict(refToPSet_ = 'lowPtQuadStepTrajectoryBuilder'),
    TrajectoryCleaner           = 'lowPtQuadStepTrajectoryCleanerBySharedHits',
    clustersToSkip              = 'lowPtQuadStepClusters',
    doSeedingRegionRebuilding   = True,
    useHitsSplitting            = True,
)
trackingPhase2PU140.toModify(lowPtQuadStepTrackCandidates,
    clustersToSkip       = '',
    phase2clustersToSkip = 'lowPtQuadStepClusters'
)

from Configuration.ProcessModifiers.trackingMkFitLowPtQuadStep_cff import trackingMkFitLowPtQuadStep
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
import RecoTracker.MkFit.mkFitIterationConfigESProducer_cfi as mkFitIterationConfigESProducer_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
lowPtQuadStepTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
    seeds = 'lowPtQuadStepSeeds',
)
lowPtQuadStepTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
    ComponentName = 'lowPtQuadStepTrackCandidatesMkFitConfig',
    config = 'RecoTracker/MkFit/data/mkfit-phase1-lowPtQuadStep.json',
)
lowPtQuadStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
    seeds = 'lowPtQuadStepTrackCandidatesMkFitSeeds',
    config = ('', 'lowPtQuadStepTrackCandidatesMkFitConfig'),
    clustersToSkip = 'lowPtQuadStepClusters',
)
trackingMkFitLowPtQuadStep.toReplaceWith(lowPtQuadStepTrackCandidates, mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
    seeds = 'lowPtQuadStepSeeds',
    mkFitSeeds = 'lowPtQuadStepTrackCandidatesMkFitSeeds',
    tracks = 'lowPtQuadStepTrackCandidatesMkFit',
))
(pp_on_XeXe_2017 | pp_on_AA).toModify(lowPtQuadStepTrackCandidatesMkFitConfig, minPt=0.49)

#For FastSim phase1 tracking
import FastSimulation.Tracking.TrackCandidateProducer_cfi
_fastSim_lowPtQuadStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src                      = 'lowPtQuadStepSeeds',
    MinNumberOfCrossedLayers = 3,
    hitMasks                 = cms.InputTag('lowPtQuadStepMasks')
)
fastSim.toReplaceWith(lowPtQuadStepTrackCandidates,_fastSim_lowPtQuadStepTrackCandidates)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi
lowPtQuadStepTracks = RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi.TrackProducerIterativeDefault.clone(
    src           = 'lowPtQuadStepTrackCandidates',
    AlgorithmName = 'lowPtQuadStep',
    Fitter        = 'FlexibleKFFittingSmoother',
)
fastSim.toModify(lowPtQuadStepTracks,TTRHBuilder = 'WithoutRefit')

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify(lowPtQuadStepTracks, TrajectoryInEvent = True)

# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
lowPtQuadStep = TrackMVAClassifierPrompt.clone(
     mva         = dict(GBRForestLabel = 'MVASelectorLowPtQuadStep_Phase1'),
     src         = 'lowPtQuadStepTracks',
     qualityCuts = [-0.7,-0.35,-0.15]
)

from RecoTracker.FinalTrackSelectors.trackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_CKF_cfi import *
trackdnn.toReplaceWith(lowPtQuadStep, trackTfClassifier.clone(
    src = 'lowPtQuadStepTracks',
    qualityCuts = qualityCutDictionary.LowPtQuadStep.value()
))
highBetaStar_2018.toModify(lowPtQuadStep,qualityCuts = [-0.9,-0.35,-0.15])
pp_on_AA.toModify(lowPtQuadStep, 
        mva         = dict(GBRForestLabel = 'HIMVASelectorLowPtQuadStep_Phase1'),
        qualityCuts = [-0.9, -0.4, 0.3],
)
fastSim.toModify(lowPtQuadStep,vertices = 'firstStepPrimaryVerticesBeforeMixing')

((~trackingMkFitLowPtQuadStep) & trackdnn).toModify(lowPtQuadStep, mva = dict(tfDnnLabel  = 'trackSelectionTf_CKF'),
                                                    qualityCuts = [-0.33,  0.13,  0.35])

# For Phase2PU140
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtQuadStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'lowPtQuadStepTracks',
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtQuadStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.6, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtQuadStepTight',
            preFilterName = 'lowPtQuadStepLoose',
            chi2n_par = 1.4,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.6, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtQuadStep',
            preFilterName = 'lowPtQuadStepTight',
            min_eta = -4.0,
            max_eta = 4.0,
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.5, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.45, 4.0 )
            ),
    ] #end of vpset
) #end of clone


# Final sequence
LowPtQuadStepTask = cms.Task(lowPtQuadStepClusters,
                             lowPtQuadStepSeedLayers,
                             lowPtQuadStepTrackingRegions,
                             lowPtQuadStepHitDoublets,
                             lowPtQuadStepHitQuadruplets,
                             lowPtQuadStepSeeds,
                             lowPtQuadStepTrackCandidates,
                             lowPtQuadStepTracks,
                             lowPtQuadStep)
LowPtQuadStep = cms.Sequence(LowPtQuadStepTask)

_LowPtQuadStepTask_trackingMkFit = LowPtQuadStepTask.copy()
_LowPtQuadStepTask_trackingMkFit.add(lowPtQuadStepTrackCandidatesMkFitSeeds, lowPtQuadStepTrackCandidatesMkFit, lowPtQuadStepTrackCandidatesMkFitConfig)
trackingMkFitLowPtQuadStep.toReplaceWith(LowPtQuadStepTask, _LowPtQuadStepTask_trackingMkFit)

_LowPtQuadStepTask_Phase2PU140 = LowPtQuadStepTask.copy()
_LowPtQuadStepTask_Phase2PU140.replace(lowPtQuadStep, lowPtQuadStepSelector)
trackingPhase2PU140.toReplaceWith(LowPtQuadStepTask, _LowPtQuadStepTask_Phase2PU140)

# fast tracking mask producer
from FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi import maskProducerFromClusterRemover
lowPtQuadStepMasks = maskProducerFromClusterRemover(lowPtQuadStepClusters)
fastSim.toReplaceWith(LowPtQuadStepTask,
                      cms.Task(lowPtQuadStepMasks
                               ,lowPtQuadStepTrackingRegions
                               ,lowPtQuadStepSeeds
                               ,lowPtQuadStepTrackCandidates
                               ,lowPtQuadStepTracks
                               ,lowPtQuadStep
                               ) )
