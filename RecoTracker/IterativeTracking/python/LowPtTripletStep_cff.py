import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
from Configuration.Eras.Modifier_fastSim_cff import fastSim

#for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

# NEW CLUSTERS (remove previously used clusters)
lowPtTripletStepClusters = _cfg.clusterRemoverForIter('LowPtTripletStep')
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(lowPtTripletStepClusters, _cfg.clusterRemoverForIter('LowPtTripletStep', _eraName, _postfix))

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix = dict(skipClusters = cms.InputTag('lowPtTripletStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('lowPtTripletStepClusters'))
)
_layerListForPhase1 = [
    'BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
    'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
    'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
    'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
    'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
    'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
    'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
    'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
    'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
    'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'
]
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(lowPtTripletStepSeedLayers, layerList = _layerListForPhase1)

# combination with gap removed as only source of fakes in current geometry (kept for doc,=)
_layerListForPhase2 = ['BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
#                       'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
                       'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                       'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
#                       'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
                       'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
#                       'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg',
                       'FPix2_pos+FPix3_pos+FPix4_pos', 'FPix2_neg+FPix3_neg+FPix4_neg',
                       'FPix3_pos+FPix4_pos+FPix5_pos', 'FPix3_neg+FPix4_neg+FPix5_neg',
                       'FPix4_pos+FPix5_pos+FPix6_pos', 'FPix4_neg+FPix5_neg+FPix6_neg',
#  removed as redunant and covering effectively only eta>4   (here for documentation, to be optimized after TDR)
#                       'FPix5_pos+FPix6_pos+FPix7_pos', 'FPix5_neg+FPix6_neg+FPix7_neg',
#                       'FPix6_pos+FPix7_pos+FPix8_pos', 'FPix6_neg+FPix7_neg+FPix8_neg'
]
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(lowPtTripletStepSeedLayers, layerList = _layerListForPhase2)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
lowPtTripletStepTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin        = 0.2,
    originRadius = 0.02,
    nSigmaZ      = 4.0
))
trackingPhase1.toModify(lowPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.2))
trackingPhase2PU140.toModify(lowPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.40))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
(pp_on_XeXe_2017 | pp_on_AA).toReplaceWith(lowPtTripletStepTrackingRegions, 
                _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
                    useFixedError = False,
                    ptMin         = 0.49,
                    originRadius  = 0.02 )
               )
)
from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(lowPtTripletStepTrackingRegions,RegionPSet = dict(
     ptMin        = 0.05,
     originRadius = 0.2 )
)

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
lowPtTripletStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers   = 'lowPtTripletStepSeedLayers',
    trackingRegions = 'lowPtTripletStepTrackingRegions',
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtTripletStepHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets              = 'lowPtTripletStepHitDoublets',
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
lowPtTripletStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = 'lowPtTripletStepHitTriplets',
)

from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
trackingPhase1.toModify(lowPtTripletStepHitDoublets, layerPairs = [0,1]) # layer pairs (0,1), (1,2)
trackingPhase1.toReplaceWith(lowPtTripletStepHitTriplets, _caHitTripletEDProducer.clone(
    doublets = 'lowPtTripletStepHitDoublets',
    extraHitRPhitolerance = lowPtTripletStepHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = lowPtTripletStepHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 70 , value2 = 8,
    ),
    useBendingCorrection = True,
    CAThetaCut           = 0.002,
    CAPhiCut             = 0.05 )
)

trackingPhase2PU140.toModify(lowPtTripletStepHitDoublets, layerPairs = [0,1]) # layer pairs (0,1), (1,2)
trackingPhase2PU140.toReplaceWith(lowPtTripletStepHitTriplets, _caHitTripletEDProducer.clone(
    doublets = 'lowPtTripletStepHitDoublets',
    extraHitRPhitolerance = lowPtTripletStepHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = lowPtTripletStepHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 70 , value2 = 8,
    ),
    useBendingCorrection = True,
    CAThetaCut           = 0.002,
    CAPhiCut             = 0.05 )
)
highBetaStar_2018.toModify(lowPtTripletStepHitTriplets,CAThetaCut = 0.004,CAPhiCut = 0.1)
 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet
_fastSim_lowPtTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = 'lowPtTripletStepTrackingRegions',
    hitMasks        = cms.InputTag('lowPtTripletStepMasks'),
    seedFinderSelector = dict(pixelTripletGeneratorFactory = _hitSetProducerToFactoryPSet(lowPtTripletStepHitTriplets).clone(
                              SeedComparitorPSet = dict(ComponentName = 'none')),
                              layerList = lowPtTripletStepSeedLayers.layerList.value()
                         )
)
#new for phase1
trackingPhase1.toModify(_fastSim_lowPtTripletStepSeeds, seedFinderSelector = dict(
        pixelTripletGeneratorFactory = None,
        CAHitTripletGeneratorFactory = _hitSetProducerToFactoryPSet(lowPtTripletStepHitTriplets).clone(SeedComparitorPSet = dict(ComponentName = 'none')),
        #new parameters required for phase1 seeding
        BPix = dict(
            TTRHBuilder = 'WithoutRefit',
            HitProducer = 'TrackingRecHitProducer',
            ),
        FPix = dict(
            TTRHBuilder = 'WithoutRefit',
            HitProducer = 'TrackingRecHitProducer',
            ),
        layerPairs = lowPtTripletStepHitDoublets.layerPairs.value()
        )
)
fastSim.toReplaceWith(lowPtTripletStepSeeds,_fastSim_lowPtTripletStepSeeds)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_lowPtTripletStepStandardTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt               = 0.075,
)
lowPtTripletStepStandardTrajectoryFilter = _lowPtTripletStepStandardTrajectoryFilterBase.clone(
    maxCCCLostHits     = 0,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(lowPtTripletStepStandardTrajectoryFilter, maxCCCLostHits = 1)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)
trackingPhase2PU140.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)

(pp_on_XeXe_2017 | pp_on_AA).toModify(lowPtTripletStepStandardTrajectoryFilter, minPt=0.49)

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtTripletStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters   = [cms.PSet(refToPSet_ = cms.string('lowPtTripletStepStandardTrajectoryFilter')),
                 # cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))
                ]
    )
trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryFilter,
    filters = lowPtTripletStepTrajectoryFilter.filters + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

lowPtTripletStepTrajectoryFilterInOut = lowPtTripletStepStandardTrajectoryFilter.clone(
    minimumNumberOfHits = 4,
    seedExtension       = 1,
    strictSeedExtension = False, # allow inactive
    pixelSeedExtension  = False,
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName    = 'lowPtTripletStepChi2Est',
    nSigma           = 3.0,
    MaxChi2          = 9.0,
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
)
_tracker_apv_vfp30_2016.toModify(lowPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter       = cms.PSet(refToPSet_ = cms.string('lowPtTripletStepTrajectoryFilter')),
    maxCand                = 4,
    estimator              = 'lowPtTripletStepChi2Est',
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
)
trackingLowPU.toModify(lowPtTripletStepTrajectoryBuilder, maxCand = 3)
trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryBuilder, 
    inOutTrajectoryFilter = dict(refToPSet_ = 'lowPtTripletStepTrajectoryFilterInOut'),
    useSameTrajFilter     = False,
    maxCand               = 3,
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'lowPtTripletStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner       = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet       = cms.PSet(refToPSet_ = cms.string('lowPtTripletStepTrajectoryBuilder')),
    clustersToSkip              = cms.InputTag('lowPtTripletStepClusters'),
    doSeedingRegionRebuilding   = True,
    useHitsSplitting            = True,
    TrajectoryCleaner           = 'lowPtTripletStepTrajectoryCleanerBySharedHits'
)

trackingPhase2PU140.toModify(lowPtTripletStepTrackCandidates,
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag('lowPtTripletStepClusters')
)

from Configuration.ProcessModifiers.trackingMkFitLowPtTripletStep_cff import trackingMkFitLowPtTripletStep
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitIterationConfigESProducer_cfi as mkFitIterationConfigESProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
lowPtTripletStepTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
    seeds = 'lowPtTripletStepSeeds',
)
lowPtTripletStepTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
    ComponentName = 'lowPtTripletStepTrackCandidatesMkFitConfig',
    config = 'RecoTracker/MkFit/data/mkfit-phase1-lowPtTripletStep.json',
)
lowPtTripletStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
    seeds = 'lowPtTripletStepTrackCandidatesMkFitSeeds',
    config = ('', 'lowPtTripletStepTrackCandidatesMkFitConfig'),
    clustersToSkip = 'lowPtTripletStepClusters',
)
trackingMkFitLowPtTripletStep.toReplaceWith(lowPtTripletStepTrackCandidates, mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
    seeds = 'lowPtTripletStepSeeds',
    mkFitSeeds = 'lowPtTripletStepTrackCandidatesMkFitSeeds',
    tracks = 'lowPtTripletStepTrackCandidatesMkFit',
))

import FastSimulation.Tracking.TrackCandidateProducer_cfi
fastSim.toReplaceWith(lowPtTripletStepTrackCandidates,
                      FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
        src = 'lowPtTripletStepSeeds',
        MinNumberOfCrossedLayers = 3,
        hitMasks = cms.InputTag('lowPtTripletStepMasks'))
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src           = 'lowPtTripletStepTrackCandidates',
    AlgorithmName = 'lowPtTripletStep',
    Fitter        = 'FlexibleKFFittingSmoother'
)
fastSim.toModify(lowPtTripletStepTracks, TTRHBuilder = 'WithoutRefit')

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
lowPtTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName       = 'lowPtTripletStepTrajectoryCleanerBySharedHits',
    fractionShared      = 0.16,
    allowSharedFirstHit = True
)
trackingLowPU.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.19)
trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)

# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
lowPtTripletStep =  TrackMVAClassifierPrompt.clone(
    src         = 'lowPtTripletStepTracks',
    mva         = dict(GBRForestLabel = 'MVASelectorIter1_13TeV'),
    qualityCuts = [-0.6,-0.3,-0.1]
)
trackingPhase1.toReplaceWith(lowPtTripletStep, lowPtTripletStep.clone(
     mva         = dict(GBRForestLabel = 'MVASelectorLowPtTripletStep_Phase1'),
     qualityCuts = [-0.4,0.0,0.3],
))

from RecoTracker.FinalTrackSelectors.TrackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
trackdnn.toReplaceWith(lowPtTripletStep, TrackTfClassifier.clone(
    src = 'lowPtTripletStepTracks',
    qualityCuts = qualityCutDictionary['LowPtTripletStep']
))

highBetaStar_2018.toModify(lowPtTripletStep,qualityCuts = [-0.7,-0.3,-0.1])
pp_on_AA.toModify(lowPtTripletStep, 
        mva         = dict(GBRForestLabel = 'HIMVASelectorLowPtTripletStep_Phase1'),
        qualityCuts = [-0.8, -0.4, 0.5],
)
fastSim.toModify(lowPtTripletStep, vertices = 'firstStepPrimaryVerticesBeforeMixing')

# For LowPU and Phase2PU140
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'lowPtTripletStepTracks',
    useAnyMVA      = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter1'),
    trackSelectors= [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtTripletStepLoose',
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtTripletStepTight',
            preFilterName = 'lowPtTripletStepLoose',
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'lowPtTripletStepTight',
        ),
    ] #end of vpset
) #end of clone
trackingPhase2PU140.toModify(lowPtTripletStepSelector,
    useAnyMVA      = None,
    GBRForestLabel = None,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtTripletStepLoose',
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.6, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtTripletStepTight',
            preFilterName = 'lowPtTripletStepLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.6, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtTripletStep',
            preFilterName = 'lowPtTripletStepTight',
            min_eta = -4.0,
            max_eta = 4.0,
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            min_nhits = 3,
            minNumberLayers = 4,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 4,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.5, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.45, 4.0 )
            ),
        ), #end of vpset
) #end of clone


from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
vectorHits.toModify(lowPtTripletStepSelector.trackSelectors[2], minNumberLayers = 3, minNumber3DLayers = 3)


# Final sequence
LowPtTripletStepTask = cms.Task(lowPtTripletStepClusters,
                                lowPtTripletStepSeedLayers,
                                lowPtTripletStepTrackingRegions,
                                lowPtTripletStepHitDoublets,
                                lowPtTripletStepHitTriplets,
                                lowPtTripletStepSeeds,
                                lowPtTripletStepTrackCandidates,
                                lowPtTripletStepTracks,
                                lowPtTripletStep)
LowPtTripletStep = cms.Sequence(LowPtTripletStepTask)

_LowPtTripletStepTask_trackingMkFit = LowPtTripletStepTask.copy()
_LowPtTripletStepTask_trackingMkFit.add(lowPtTripletStepTrackCandidatesMkFitSeeds, lowPtTripletStepTrackCandidatesMkFit)
trackingMkFitLowPtTripletStep.toReplaceWith(LowPtTripletStepTask, _LowPtTripletStepTask_trackingMkFit)

_LowPtTripletStepTask_LowPU_Phase2PU140 = LowPtTripletStepTask.copy()
_LowPtTripletStepTask_LowPU_Phase2PU140.replace(lowPtTripletStep, lowPtTripletStepSelector)
trackingLowPU.toReplaceWith(LowPtTripletStepTask, _LowPtTripletStepTask_LowPU_Phase2PU140)

trackingPhase2PU140.toReplaceWith(LowPtTripletStepTask, _LowPtTripletStepTask_LowPU_Phase2PU140)
#fastsim
from FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi import maskProducerFromClusterRemover
lowPtTripletStepMasks = maskProducerFromClusterRemover(lowPtTripletStepClusters)
fastSim.toReplaceWith(LowPtTripletStepTask,
                      cms.Task(lowPtTripletStepMasks
                                   ,lowPtTripletStepTrackingRegions
                                   ,lowPtTripletStepSeeds
                                   ,lowPtTripletStepTrackCandidates
                                   ,lowPtTripletStepTracks  
                                   ,lowPtTripletStep   
                                   ))
