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
                 'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
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
                 'FPix6_pos+FPix7_pos+FPix8_pos', 'FPix6_neg+FPix7_neg+FPix8_neg',
#  removed as redunant and covering effectively only eta>4   (here for documentation, to be optimized after TDR)
#                 'FPix6_pos+FPix7_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix9_neg']
     ]
)
# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
highPtTripletStepTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 0.55,
    originRadius = 0.02,
    nSigmaZ = 4.0
))
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1QuadProp.toModify(highPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.6))
trackingPhase2PU140.toModify(highPtTripletStepTrackingRegions, RegionPSet = dict(ptMin = 0.7, originRadius = 0.02))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
pp_on_XeXe_2017.toReplaceWith(highPtTripletStepTrackingRegions, 
                              _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
            fixedError = 0.2,
            ptMin = 0.6,
            originRadius = 0.02
            )
                                                                      )
)


# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
highPtTripletStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "highPtTripletStepSeedLayers",
    trackingRegions = "highPtTripletStepTrackingRegions",
    layerPairs = [0,1], # layer pairs (0,1), (1,2)
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
highPtTripletStepHitTriplets = _caHitTripletEDProducer.clone(
    doublets = "highPtTripletStepHitDoublets",
    extraHitRPhitolerance = _pixelTripletHLTEDProducer.extraHitRPhitolerance,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 8,
        value1 = 100, value2 = 6,
    ),
    useBendingCorrection = True,
    CAThetaCut = 0.004,
    CAPhiCut = 0.07,
    CAHardPtCut = 0.3,
)
trackingPhase2PU140.toModify(highPtTripletStepHitTriplets,CAThetaCut = 0.003,CAPhiCut = 0.06,CAHardPtCut = 0.5)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
highPtTripletStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "highPtTripletStepHitTriplets",
)

trackingPhase1QuadProp.toModify(highPtTripletStepHitDoublets, layerPairs = [0]) # layer pair (0,1)
_highPtTripletStepHitTriplets_propagation = _pixelTripletHLTEDProducer.clone(
    doublets = "highPtTripletStepHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = highPtTripletStepHitTriplets.SeedComparitorPSet,
)
trackingPhase1QuadProp.toReplaceWith(highPtTripletStepHitTriplets, _highPtTripletStepHitTriplets_propagation)

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
trackingPhase2PU140.toReplaceWith(highPtTripletStepTrajectoryFilterBase, _highPtTripletStepTrajectoryFilterBase)

pp_on_XeXe_2017.toModify(highPtTripletStepTrajectoryFilterBase, minPt=0.7)

highPtTripletStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('highPtTripletStepTrajectoryFilterBase'))]
)

trackingPhase2PU140.toModify(highPtTripletStepTrajectoryFilter,
    filters = highPtTripletStepTrajectoryFilter.filters + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)


highPtTripletStepTrajectoryFilterInOut = highPtTripletStepTrajectoryFilterBase.clone(
    minPt = 0.4,
    minimumNumberOfHits = 4,
    seedExtension = 1,
    strictSeedExtension = False, # allow inactive
    pixelSeedExtension = False,
)


import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
highPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'highPtTripletStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 30.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutLoose'),
    pTChargeCutThreshold = 15.
)
trackingPhase2PU140.toModify(highPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutNone"),
    MaxChi2 = cms.double(20.0)
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
trackingPhase2PU140.toModify(highPtTripletStepTrajectoryBuilder,
    inOutTrajectoryFilter = dict(refToPSet_ = "highPtTripletStepTrajectoryFilterInOut"),
    useSameTrajFilter = False,
    maxCand = 3,
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

# For Phase2PU140
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits as _trajectoryCleanerBySharedHits
highPtTripletStepTrajectoryCleanerBySharedHits = _trajectoryCleanerBySharedHits.clone(
    ComponentName = 'highPtTripletStepTrajectoryCleanerBySharedHits',
    fractionShared = 0.16,
    allowSharedFirstHit = True
)
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
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
highPtTripletStep = TrackMVAClassifierPrompt.clone(
    src	= 'highPtTripletStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorHighPtTripletStep_Phase1'),
    qualityCuts	= [0.2,0.3,0.4],
)


# For Phase2PU140
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
            dz_par2 = ( 0.6, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'highPtTripletStep',
            preFilterName = 'highPtTripletStepTight',
            chi2n_par = 0.8,
            res_par = ( 0.003, 0.001 ),
            min_nhits = 4,
            minNumberLayers = 4,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 4,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.55, 4.0 )
            ),
    ] #end of vpset
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
_HighPtTripletStep_Phase2PU140 = HighPtTripletStep.copy()
_HighPtTripletStep_Phase2PU140.replace(highPtTripletStep, highPtTripletStepSelector)
trackingPhase2PU140.toReplaceWith(HighPtTripletStep, _HighPtTripletStep_Phase2PU140)
