import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

# NEW CLUSTERS (remove previously used clusters)
lowPtTripletStepClusters = _cfg.clusterRemoverForIter("LowPtTripletStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(lowPtTripletStepClusters, _cfg.clusterRemoverForIter("LowPtTripletStep", _eraName, _postfix))

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
lowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')
lowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')
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
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1.toModify(lowPtTripletStepSeedLayers, layerList = _layerListForPhase1)
trackingPhase1QuadProp.toModify(lowPtTripletStepSeedLayers, layerList = _layerListForPhase1)
trackingPhase1PU70.toModify(lowPtTripletStepSeedLayers, layerList = _layerListForPhase1)

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
    ptMin = 0.2,
    originRadius = 0.02,
    nSigmaZ = 4.0
))
trackingPhase1.toModify(lowPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.2))
trackingPhase1QuadProp.toModify(lowPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.35)) # FIXME: Phase1PU70 value, let's see if we can lower it to Run2 value (0.2)
trackingPhase1PU70.toModify(lowPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.35, originRadius = 0.015))
trackingPhase2PU140.toModify(lowPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.45))

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
lowPtTripletStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "lowPtTripletStepSeedLayers",
    trackingRegions = "lowPtTripletStepTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtTripletStepHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "lowPtTripletStepHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
lowPtTripletStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "lowPtTripletStepHitTriplets",
)

from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
trackingPhase1.toModify(lowPtTripletStepHitDoublets, layerPairs = [0,1]) # layer pairs (0,1), (1,2)
trackingPhase1.toReplaceWith(lowPtTripletStepHitTriplets, _caHitTripletEDProducer.clone(
    doublets = "lowPtTripletStepHitDoublets",
    extraHitRPhitolerance = lowPtTripletStepHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = lowPtTripletStepHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 70 , value2 = 8,
    ),
    useBendingCorrection = True,
    CAThetaCut = 0.002,
    CAPhiCut = 0.05,
))


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_lowPtTripletStepStandardTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
)
lowPtTripletStepStandardTrajectoryFilter = _lowPtTripletStepStandardTrajectoryFilterBase.clone(
    maxCCCLostHits = 0,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(lowPtTripletStepStandardTrajectoryFilter, maxCCCLostHits = 1)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)
trackingPhase1PU70.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)
trackingPhase2PU140.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtTripletStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters   = [cms.PSet(refToPSet_ = cms.string('lowPtTripletStepStandardTrajectoryFilter')),
                 # cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))
                ]
    )
trackingPhase1PU70.toModify(lowPtTripletStepTrajectoryFilter,
    filters = lowPtTripletStepTrajectoryFilter.filters + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)
trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryFilter,
    filters = lowPtTripletStepTrajectoryFilter.filters + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('lowPtTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
)
_tracker_apv_vfp30_2016.toModify(lowPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutTiny")
)
trackingPhase1PU70.toModify(lowPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('lowPtTripletStepTrajectoryFilter')),
    maxCand = 4,
    estimator = cms.string('lowPtTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
    )
trackingLowPU.toModify(lowPtTripletStepTrajectoryBuilder, maxCand = 3)
trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryBuilder, maxCand = 3)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('lowPtTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('lowPtTripletStepTrajectoryBuilder')),
    clustersToSkip = cms.InputTag('lowPtTripletStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
trackingPhase2PU140.toModify(lowPtTripletStepTrackCandidates,
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag("lowPtTripletStepClusters")
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtTripletStepTrackCandidates',
    AlgorithmName = cms.string('lowPtTripletStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
lowPtTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('lowPtTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.16),
            allowSharedFirstHit = cms.bool(True)
            )
lowPtTripletStepTrackCandidates.TrajectoryCleaner = 'lowPtTripletStepTrajectoryCleanerBySharedHits'
trackingLowPU.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.19)
trackingPhase1PU70.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)
trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)

# Final selection



from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
lowPtTripletStep =  TrackMVAClassifierPrompt.clone()
lowPtTripletStep.src = 'lowPtTripletStepTracks'
lowPtTripletStep.GBRForestLabel = 'MVASelectorIter1_13TeV'
lowPtTripletStep.qualityCuts = [-0.6,-0.3,-0.1]

# For Phase1
# MVA selection to be enabled after re-training, for time being we go with cut-based selector
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import TrackCutClassifier as _TrackCutClassifier
_cutClassifierForPhase1 = _TrackCutClassifier.clone(
    src = "lowPtTripletStepTracks",
    vertices = "firstStepPrimaryVertices",
    mva = dict (
        minPixelHits = [1,1,1],
        maxChi2 = [9999.,9999.,9999.],
        maxChi2n = [2.0,0.9,0.5],
        minLayers = [3,3,3],
        min3DLayers = [3,3,3],
        maxLostLayers = [2,2,2],
        dz_par = dict(
            dz_par1 = [0.7,0.6,0.45],
            dz_par2 = [0.5,0.4,0.4],
            dz_exp = [4,4,4],
        ),
        dr_par = dict(
            dr_par1 = [0.8,0.7,0.6],
            dr_par2 = [0.5,0.4,0.3],
            dr_exp = [4,4,4],
            d0err_par = [0.002,0.002,0.001],
        ),
    )
)
trackingPhase1.toReplaceWith(lowPtTripletStep, _cutClassifierForPhase1)
trackingPhase1QuadProp.toReplaceWith(lowPtTripletStep, _cutClassifierForPhase1)

# For LowPU and Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'lowPtTripletStepTracks',
    useAnyMVA = cms.bool(False),
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
trackingPhase1PU70.toModify(lowPtTripletStepSelector,
    useAnyMVA = None,
    GBRForestLabel = None,
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtTripletStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtTripletStepTight',
            preFilterName = 'lowPtTripletStepLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.6, 4.0 ),
            d0_par2 = ( 0.4, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtTripletStep',
            preFilterName = 'lowPtTripletStepTight',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.45, 4.0 ),
            d0_par2 = ( 0.3, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
    ] #end of vpset
) #end of clone

trackingPhase2PU140.toModify(lowPtTripletStepSelector,
    useAnyMVA = None,
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
            dz_par1 = ( 0.6, 4.0 ),
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
            dz_par1 = ( 0.5, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtTripletStep',
            preFilterName = 'lowPtTripletStepTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.4, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.45, 4.0 )
            ),
        ), #end of vpset
    vertices = "pixelVertices"
) #end of clone



# Final sequence
LowPtTripletStep = cms.Sequence(lowPtTripletStepClusters*
                                lowPtTripletStepSeedLayers*
                                lowPtTripletStepTrackingRegions*
                                lowPtTripletStepHitDoublets*
                                lowPtTripletStepHitTriplets*
                                lowPtTripletStepSeeds*
                                lowPtTripletStepTrackCandidates*
                                lowPtTripletStepTracks*
                                lowPtTripletStep)
_LowPtTripletStep_LowPU = LowPtTripletStep.copy()
_LowPtTripletStep_LowPU.replace(lowPtTripletStep, lowPtTripletStepSelector)
trackingLowPU.toReplaceWith(LowPtTripletStep, _LowPtTripletStep_LowPU)
_LowPtTripletStep_Phase1PU70 = LowPtTripletStep.copy()
_LowPtTripletStep_Phase1PU70.replace(lowPtTripletStep, lowPtTripletStepSelector)
trackingPhase1PU70.toReplaceWith(LowPtTripletStep, _LowPtTripletStep_Phase1PU70)
trackingPhase2PU140.toReplaceWith(LowPtTripletStep, _LowPtTripletStep_Phase1PU70)
