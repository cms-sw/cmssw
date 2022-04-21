import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

#for fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet

#for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

# for no-loopers
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

###############################################
# Low pT and detached tracks from pixel triplets
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS
detachedTripletStepClusters = _cfg.clusterRemoverForIter('DetachedTripletStep')
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(detachedTripletStepClusters, _cfg.clusterRemoverForIter('DetachedTripletStep', _eraName, _postfix))

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
detachedTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix = dict(skipClusters = cms.InputTag('detachedTripletStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('detachedTripletStepClusters'))
)
_phase1LayerList = [
        'BPix1+BPix2+BPix3',
        'BPix2+BPix3+BPix4',
#        'BPix1+BPix3+BPix4', # has 'hole', not tested
#        'BPix1+BPix2+BPix4', # has 'hole', not tested
        'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
#        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', # mostly fake tracks, lots of seeds
#        'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',  # has 'hole', not tested
        'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
#        'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg', # mostly fake tracks, lots of seeds
#        'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',  # has 'hole', not tested
        'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
#        'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',  # has 'hole', not tested
#        'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'  # has 'hole', not tested
    ]
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(detachedTripletStepSeedLayers, layerList=_phase1LayerList)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpotFixedZ_cfi import globalTrackingRegionFromBeamSpotFixedZ as _globalTrackingRegionFromBeamSpotFixedZ
detachedTripletStepTrackingRegions = _globalTrackingRegionFromBeamSpotFixedZ.clone(RegionPSet = dict(
    ptMin            = 0.3,
    originHalfLength = 15.0,
    originRadius     = 1.5
))
trackingPhase1.toModify(detachedTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.25))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
(pp_on_XeXe_2017 | pp_on_AA).toReplaceWith(detachedTripletStepTrackingRegions, 
                _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
                    fixedError   = 2.5,
                    ptMin        = 0.9,
                    originRadius = 1.5)
                )
)
from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(detachedTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.05))

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
detachedTripletStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers   = 'detachedTripletStepSeedLayers',
    trackingRegions = 'detachedTripletStepTrackingRegions',
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletLargeTipEDProducer_cfi import pixelTripletLargeTipEDProducer as _pixelTripletLargeTipEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
detachedTripletStepHitTriplets = _pixelTripletLargeTipEDProducer.clone(
    doublets = 'detachedTripletStepHitDoublets',
    produceSeedingHitSets = True,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
detachedTripletStepSeeds = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets = 'detachedTripletStepHitTriplets',
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName      = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits    = cms.bool(True),
        FilterStripHits    = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    ),
)

from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
trackingPhase1.toModify(detachedTripletStepHitDoublets, layerPairs = [0,1]) # layer pairs (0,1), (1,2)
trackingPhase1.toReplaceWith(detachedTripletStepHitTriplets, _caHitTripletEDProducer.clone(
    doublets = 'detachedTripletStepHitDoublets',
    extraHitRPhitolerance = detachedTripletStepHitTriplets.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 300, value2 = 10,
    ),
    useBendingCorrection = True,
    CAThetaCut           = 0.001,
    CAPhiCut             = 0,
    CAHardPtCut          = 0.2,
))
highBetaStar_2018.toModify(detachedTripletStepHitTriplets,CAThetaCut = 0.002,CAPhiCut = 0.1,CAHardPtCut = 0)

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
_fastSim_detachedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = 'detachedTripletStepTrackingRegions',
    hitMasks        = cms.InputTag('detachedTripletStepMasks'),
    seedFinderSelector = dict( pixelTripletGeneratorFactory = _hitSetProducerToFactoryPSet(detachedTripletStepHitTriplets),
                               layerList = detachedTripletStepSeedLayers.layerList.value())
)#new for phase1
trackingPhase1.toModify(_fastSim_detachedTripletStepSeeds, seedFinderSelector = dict(
        pixelTripletGeneratorFactory = None,
        CAHitTripletGeneratorFactory = _hitSetProducerToFactoryPSet(detachedTripletStepHitTriplets),
        #new parameters required for phase1 seeding
        BPix = dict(
            TTRHBuilder = 'WithoutRefit',
            HitProducer = 'TrackingRecHitProducer',
            ),
        FPix = dict(
            TTRHBuilder = 'WithoutRefit',
            HitProducer = 'TrackingRecHitProducer',
            ),
        layerPairs = detachedTripletStepHitDoublets.layerPairs.value()
        )
)
fastSim.toReplaceWith(detachedTripletStepSeeds,_fastSim_detachedTripletStepSeeds)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_detachedTripletStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
#    maxLostHitsFraction = cms.double(1./10.),
#    constantValueForLostHitsFractionFilter = cms.double(0.701),
    minimumNumberOfHits = 3,
    minPt               = 0.075,
)
detachedTripletStepTrajectoryFilterBase = _detachedTripletStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 0,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(detachedTripletStepTrajectoryFilterBase, maxCCCLostHits = 2)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(detachedTripletStepTrajectoryFilterBase, _detachedTripletStepTrajectoryFilterBase.clone(
    maxLostHitsFraction = 1./10.,
    constantValueForLostHitsFractionFilter = 0.701,
))

(pp_on_XeXe_2017 | pp_on_AA).toModify(detachedTripletStepTrajectoryFilterBase, minPt=0.9)

import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
detachedTripletStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
detachedTripletStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('detachedTripletStepTrajectoryFilterBase')),
#        cms.PSet( refToPSet_ = cms.string('detachedTripletStepTrajectoryFilterShape'))
    ),
)


import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
detachedTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'detachedTripletStepChi2Est',
    nSigma        = 3.0,
    MaxChi2       = 9.0,
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
)
_tracker_apv_vfp30_2016.toModify(detachedTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
detachedTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'detachedTripletStepTrajectoryFilter'),
    maxCand = 3,
    alwaysUseInvalidHits = True,
    estimator = 'detachedTripletStepChi2Est',
    maxDPhiForLooperReconstruction = 2.0,
    maxPtForLooperReconstruction = 0.7,
)
trackingNoLoopers.toModify(detachedTripletStepTrajectoryBuilder,
                           maxPtForLooperReconstruction = 0.0)
trackingLowPU.toModify(detachedTripletStepTrajectoryBuilder,
    maxCand = 2,
    alwaysUseInvalidHits = False,
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
# Give handle for CKF for HI
_detachedTripletStepTrackCandidatesCkf = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'detachedTripletStepSeeds',
    clustersToSkip = 'detachedTripletStepClusters',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'detachedTripletStepTrajectoryBuilder'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
detachedTripletStepTrackCandidates = _detachedTripletStepTrackCandidatesCkf.clone()

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
detachedTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName       = 'detachedTripletStepTrajectoryCleanerBySharedHits',
    fractionShared      = 0.13,
    allowSharedFirstHit = True
)
detachedTripletStepTrackCandidates.TrajectoryCleaner = 'detachedTripletStepTrajectoryCleanerBySharedHits'
trackingLowPU.toModify(detachedTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.19)

from Configuration.ProcessModifiers.trackingMkFitDetachedTripletStep_cff import trackingMkFitDetachedTripletStep
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
import RecoTracker.MkFit.mkFitIterationConfigESProducer_cfi as mkFitIterationConfigESProducer_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
detachedTripletStepTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
    seeds = 'detachedTripletStepSeeds',
)
detachedTripletStepTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
    ComponentName = 'detachedTripletStepTrackCandidatesMkFitConfig',
    config = 'RecoTracker/MkFit/data/mkfit-phase1-detachedTripletStep.json',
)
detachedTripletStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
    seeds = 'detachedTripletStepTrackCandidatesMkFitSeeds',
    config = ('', 'detachedTripletStepTrackCandidatesMkFitConfig'),
    clustersToSkip = 'detachedTripletStepClusters',
)
trackingMkFitDetachedTripletStep.toReplaceWith(detachedTripletStepTrackCandidates, mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
    seeds = 'detachedTripletStepSeeds',
    mkFitSeeds = 'detachedTripletStepTrackCandidatesMkFitSeeds',
    tracks = 'detachedTripletStepTrackCandidatesMkFit',
))
(pp_on_XeXe_2017 | pp_on_AA).toModify(detachedTripletStepTrackCandidatesMkFitConfig, minPt=0.9)

import FastSimulation.Tracking.TrackCandidateProducer_cfi
_fastSim_detachedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = 'detachedTripletStepSeeds',
    MinNumberOfCrossedLayers = 3,
    hitMasks = cms.InputTag('detachedTripletStepMasks')
)
fastSim.toReplaceWith(detachedTripletStepTrackCandidates,_fastSim_detachedTripletStepTrackCandidates)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
detachedTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = 'detachedTripletStep',
    src           = 'detachedTripletStepTrackCandidates',
    Fitter        = 'FlexibleKFFittingSmoother'
)
fastSim.toModify(detachedTripletStepTracks,TTRHBuilder = 'WithoutRefit')

# TRACK SELECTION AND QUALITY FLAG SETTING.


from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
detachedTripletStepClassifier1 = TrackMVAClassifierDetached.clone(
    src = 'detachedTripletStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorIter3_13TeV'),
    qualityCuts = [-0.5,0.0,0.5]
)
fastSim.toModify(detachedTripletStepClassifier1,vertices = 'firstStepPrimaryVerticesBeforeMixing')

detachedTripletStepClassifier2 = TrackMVAClassifierPrompt.clone(
    src = 'detachedTripletStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorIter0_13TeV'),
    qualityCuts = [-0.2,0.0,0.4]
)
fastSim.toModify(detachedTripletStepClassifier2,vertices = 'firstStepPrimaryVerticesBeforeMixing')

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
detachedTripletStep = ClassifierMerger.clone(
    inputClassifiers=['detachedTripletStepClassifier1','detachedTripletStepClassifier2']
)
trackingPhase1.toReplaceWith(detachedTripletStep, detachedTripletStepClassifier1.clone(
    mva = dict(GBRForestLabel = 'MVASelectorDetachedTripletStep_Phase1'),
    qualityCuts = [-0.2,0.3,0.8]
))

from RecoTracker.FinalTrackSelectors.trackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_CKF_cfi import *
trackdnn.toReplaceWith(detachedTripletStep, trackTfClassifier.clone(
     src = 'detachedTripletStepTracks',
     qualityCuts = qualityCutDictionary.DetachedTripletStep.value()
))
(trackdnn & fastSim).toModify(detachedTripletStep,vertices = 'firstStepPrimaryVerticesBeforeMixing')

highBetaStar_2018.toModify(detachedTripletStep,qualityCuts = [-0.5,0.0,0.5])
pp_on_AA.toModify(detachedTripletStep, 
        mva = dict(GBRForestLabel = 'HIMVASelectorDetachedTripletStep_Phase1'),
        qualityCuts = [-0.2, 0.4, 0.85],
)

# For LowPU
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'detachedTripletStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter3'),
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedTripletStepVtxLoose',
            chi2n_par = 1.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.2, 3.0 ),
            dz_par1 = ( 1.2, 3.0 ),
            d0_par2 = ( 1.3, 3.0 ),
            dz_par2 = ( 1.3, 3.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedTripletStepTrkLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.6, 4.0 ),
            dz_par1 = ( 1.6, 4.0 ),
            d0_par2 = ( 1.6, 4.0 ),
            dz_par2 = ( 1.6, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedTripletStepVtxTight',
            preFilterName = 'detachedTripletStepVtxLoose',
            chi2n_par = 0.7,
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
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedTripletStepVtx',
            preFilterName = 'detachedTripletStepVtxTight',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.85, 3.0 ),
            dz_par1 = ( 0.8, 3.0 ),
            d0_par2 = ( 0.9, 3.0 ),
            dz_par2 = ( 0.9, 3.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedTripletStepTrk',
            preFilterName = 'detachedTripletStepTrkTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 4,
            d0_par1 = ( 1.0, 4.0 ),
            dz_par1 = ( 1.0, 4.0 ),
            d0_par2 = ( 1.0, 4.0 ),
            dz_par2 = ( 1.0, 4.0 )
        )
    ] #end of vpset
) #end of clone

from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
trackingLowPU.toReplaceWith(detachedTripletStep, RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers     = ['detachedTripletStepTracks',
                          'detachedTripletStepTracks'],
    hasSelector        = [1,1],
    selectedTrackQuals = ['detachedTripletStepSelector:detachedTripletStepVtx',
                          'detachedTripletStepSelector:detachedTripletStepTrk'],
    setsToMerge        = [cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )],
    writeOnlyTrkQuals  = True
))

DetachedTripletStepTask = cms.Task(detachedTripletStepClusters,
                                   detachedTripletStepSeedLayers,
                                   detachedTripletStepTrackingRegions,
                                   detachedTripletStepHitDoublets,
                                   detachedTripletStepHitTriplets,
                                   detachedTripletStepSeeds,
                                   detachedTripletStepTrackCandidates,
                                   detachedTripletStepTracks,
                                   detachedTripletStepClassifier1,detachedTripletStepClassifier2,
                                   detachedTripletStep)
DetachedTripletStep = cms.Sequence(DetachedTripletStepTask)

_DetachedTripletStepTask_trackingMkFit = DetachedTripletStepTask.copy()
_DetachedTripletStepTask_trackingMkFit.add(detachedTripletStepTrackCandidatesMkFitSeeds, detachedTripletStepTrackCandidatesMkFit, detachedTripletStepTrackCandidatesMkFitConfig)
trackingMkFitDetachedTripletStep.toReplaceWith(DetachedTripletStepTask, _DetachedTripletStepTask_trackingMkFit)

_DetachedTripletStepTask_LowPU = DetachedTripletStepTask.copyAndExclude([detachedTripletStepClassifier2])
_DetachedTripletStepTask_LowPU.replace(detachedTripletStepClassifier1, detachedTripletStepSelector)
trackingLowPU.toReplaceWith(DetachedTripletStepTask, _DetachedTripletStepTask_LowPU)

# fast tracking mask producer
from FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi import maskProducerFromClusterRemover
detachedTripletStepMasks = maskProducerFromClusterRemover(detachedTripletStepClusters)
fastSim.toReplaceWith(DetachedTripletStepTask,
                      cms.Task(detachedTripletStepMasks
                                   ,detachedTripletStepTrackingRegions
                                   ,detachedTripletStepSeeds
                                   ,detachedTripletStepTrackCandidates
                                   ,detachedTripletStepTracks
                                   ,detachedTripletStepClassifier1,detachedTripletStepClassifier2
                                   ,detachedTripletStep
                                   ) )
