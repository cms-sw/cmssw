import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

### high-pT triplets ###

# NEW CLUSTERS (remove previously used clusters)
highPtTripletStepClusters = _cfg.clusterRemoverForIter("HighPtTripletStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(highPtTripletStepClusters, _cfg.clusterRemoverForIter("HighPtTripletStep", _eraName, _postfix))


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi as _PixelLayerTriplets_cfi
highPtTripletStepSeedLayers = _PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    layerList = [
        'BPix1+BPix2+BPix3',
        'BPix2+BPix3+BPix4',
        'BPix1+BPix3+BPix4',
        'BPix1+BPix2+BPix4',
        'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
        'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
        'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
        'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
        'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
        'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
        'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
        'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'
    ],
    BPix = dict(skipClusters = cms.InputTag('highPtTripletStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('highPtTripletStepClusters'))
)

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(highPtTripletStepSeedLayers,
# combination with gap removed as only source of fakes in current geometry (kept for doc) 
    layerList = ['BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
#                 'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
                 'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
                 'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                 'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                 'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
#                 'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
                 'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
                 'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
#                 'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg',
                 'FPix2_pos+FPix3_pos+FPix4_pos', 'FPix2_neg+FPix3_neg+FPix4_neg',
                 'FPix3_pos+FPix4_pos+FPix5_pos', 'FPix3_neg+FPix4_neg+FPix5_neg',
                 'FPix4_pos+FPix5_pos+FPix6_pos', 'FPix4_neg+FPix5_neg+FPix6_neg',
                 'FPix5_pos+FPix6_pos+FPix7_pos', 'FPix5_neg+FPix6_neg+FPix7_neg',
#  removed as redunant and covering effectively only eta>4   (here for documentation, to be optimized after TDR)
#                 'FPix6_pos+FPix7_pos+FPix8_pos', 'FPix6_neg+FPix7_neg+FPix8_neg',
#                 'FPix6_pos+FPix7_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix9_neg']
     ]
)
# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
highPtTripletStepTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 0.6,
    originRadius = 0.02,
    nSigmaZ = 4.0
))
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(highPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.55))
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toModify(highPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.7))
trackingPhase2PU140.toModify(highPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.9, originRadius = 0.03))

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
highPtTripletStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "highPtTripletStepSeedLayers",
    trackingRegions = "highPtTripletStepTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
highPtTripletStepHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "highPtTripletStepHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
highPtTripletStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "highPtTripletStepHitTriplets",
)

from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
trackingPhase1.toModify(highPtTripletStepHitDoublets, layerPairs = [0,1]) # layer pairs (0,1), (1,2)
trackingPhase1.toReplaceWith(highPtTripletStepHitTriplets, _caHitTripletEDProducer.clone(
    doublets = "highPtTripletStepHitDoublets",
    extraHitRPhitolerance = highPtTripletStepHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = highPtTripletStepHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 8,
        value1 = 100, value2 = 6,
    ),
    useBendingCorrection = True,
    CAThetaCut = 0.004,
    CAPhiCut = 0.07,
    CAHardPtCut = 0.3,
))

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_highPtTripletStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2,
)
highPtTripletStepTrajectoryFilterBase = _highPtTripletStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 0,
    minGoodStripCharge = dict(refToPSet_ = 'SiStripClusterChargeCutLoose')
)
trackingPhase1PU70.toReplaceWith(highPtTripletStepTrajectoryFilterBase, _highPtTripletStepTrajectoryFilterBase)
trackingPhase2PU140.toReplaceWith(highPtTripletStepTrajectoryFilterBase, _highPtTripletStepTrajectoryFilterBase)
highPtTripletStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('highPtTripletStepTrajectoryFilterBase'))]
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
highPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'highPtTripletStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 30.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutLoose'),
    pTChargeCutThreshold = 15.
)
trackingPhase1PU70.toModify(highPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutNone")
)
trackingPhase2PU140.toModify(highPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutNone"),
    MaxChi2 = cms.double(25.0)
)


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi as _GroupedCkfTrajectoryBuilder_cfi
highPtTripletStepTrajectoryBuilder = _GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'highPtTripletStepTrajectoryFilter'),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = 'highPtTripletStepChi2Est',
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7)
)
trackingPhase1PU70.toModify(highPtTripletStepTrajectoryBuilder,
    MeasurementTrackerName = '',
    maxCand = 4,
)
trackingPhase2PU140.toModify(highPtTripletStepTrajectoryBuilder,
    maxCand = 5,
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi as _CkfTrackCandidates_cfi
highPtTripletStepTrackCandidates = _CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'highPtTripletStepSeeds',
    clustersToSkip = cms.InputTag('highPtTripletStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = dict(refToPSet_ = 'highPtTripletStepTrajectoryBuilder'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

# For Phase1PU70 & Phase2PU140
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits as _trajectoryCleanerBySharedHits
highPtTripletStepTrajectoryCleanerBySharedHits = _trajectoryCleanerBySharedHits.clone(
    ComponentName = 'highPtTripletStepTrajectoryCleanerBySharedHits',
    fractionShared = 0.16,
    allowSharedFirstHit = True
)
trackingPhase1PU70.toModify(highPtTripletStepTrackCandidates, TrajectoryCleaner = 'highPtTripletStepTrajectoryCleanerBySharedHits')
trackingPhase2PU140.toModify(highPtTripletStepTrackCandidates, 
    TrajectoryCleaner = 'highPtTripletStepTrajectoryCleanerBySharedHits', 
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag("highPtTripletStepClusters")
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
highPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'highPtTripletStepTrackCandidates',
    AlgorithmName = 'highPtTripletStep',
    Fitter = 'FlexibleKFFittingSmoother',
)

# Final selection
# MVA selection to be enabled after re-training, for time being we go with cut-based selector
#from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
#from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
#
#highPtTripletStepClassifier1 = TrackMVAClassifierPrompt.clone()
#highPtTripletStepClassifier1.src = 'highPtTripletStepTracks'
#highPtTripletStepClassifier1.GBRForestLabel = 'MVASelectorIter0_13TeV'
#highPtTripletStepClassifier1.qualityCuts = [-0.9,-0.8,-0.7]
#
#from RecoTracker.IterativeTracking.Phase1_DetachedTripletStep_cff import detachedTripletStepClassifier1
#from RecoTracker.IterativeTracking.Phase1_LowPtTripletStep_cff import lowPtTripletStep
#highPtTripletStepClassifier2 = detachedTripletStepClassifier1.clone()
#highPtTripletStepClassifier2.src = 'highPtTripletStepTracks'
#highPtTripletStepClassifier3 = lowPtTripletStep.clone()
#highPtTripletStepClassifier3.src = 'highPtTripletStepTracks'
#
#
#from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
#highPtTripletStep = ClassifierMerger.clone()
#highPtTripletStep.inputClassifiers=['highPtTripletStepClassifier1','highPtTripletStepClassifier2','highPtTripletStepClassifier3']
#highPtTripletStep.inputClassifiers=['highPtTripletStepClassifier1','highPtTripletStepClassifier2','highPtTripletStepClassifier3']

from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import TrackCutClassifier
highPtTripletStep = TrackCutClassifier.clone(
    src = "highPtTripletStepTracks",
    vertices = "firstStepPrimaryVertices",
    mva = dict(
        minPixelHits = [1,1,1],
        maxChi2 = [9999.,9999.,9999.],
        maxChi2n = [2.0,1.0,0.7],
        minLayers = [3,3,3],
        min3DLayers = [3,3,3],
        maxLostLayers = [3,2,2],
        dz_par = dict(
            dz_par1 = [0.8,0.7,0.7],
            dz_par2 = [0.6,0.5,0.4],
            dz_exp = [4,4,4]
        ),
        dr_par = dict(
            dr_par1 = [0.7,0.6,0.5],
            dr_par2 = [0.4,0.35,0.25],
            dr_exp = [4,4,4],
            d0err_par = [0.002,0.002,0.001]
        )
    )
)

# For Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
highPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'highPtTripletStepTracks',
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'highPtTripletStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.4, 4.0 ),
            dz_par2 = ( 0.6, 4.0 )
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'highPtTripletStepTight',
            preFilterName = 'highPtTripletStepLoose',
            chi2n_par = 1.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.35, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'highPtTripletStep',
            preFilterName = 'highPtTripletStepTight',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.25, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
    ]
) #end of clone

trackingPhase2PU140.toModify(highPtTripletStepSelector,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'highPtTripletStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.6, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'highPtTripletStepTight',
            preFilterName = 'highPtTripletStepLoose',
            chi2n_par = 1.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'highPtTripletStep',
            preFilterName = 'highPtTripletStepTight',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.6, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.45, 4.0 )
            ),
        ), #end of vpset
    vertices = "pixelVertices"
) #end of clone

# Final sequence
HighPtTripletStep = cms.Sequence(highPtTripletStepClusters*
                                 highPtTripletStepSeedLayers*
                                 highPtTripletStepTrackingRegions*
                                 highPtTripletStepHitDoublets*
                                 highPtTripletStepHitTriplets*
                                 highPtTripletStepSeeds*
                                 highPtTripletStepTrackCandidates*
                                 highPtTripletStepTracks*
#                                 highPtTripletStepClassifier1*highPtTripletStepClassifier2*highPtTripletStepClassifier3*
                                 highPtTripletStep)
_HighPtTripletStep_Phase1PU70 = HighPtTripletStep.copy()
_HighPtTripletStep_Phase1PU70.replace(highPtTripletStep, highPtTripletStepSelector)
trackingPhase1PU70.toReplaceWith(HighPtTripletStep, _HighPtTripletStep_Phase1PU70)
trackingPhase2PU140.toReplaceWith(HighPtTripletStep, _HighPtTripletStep_Phase1PU70)
