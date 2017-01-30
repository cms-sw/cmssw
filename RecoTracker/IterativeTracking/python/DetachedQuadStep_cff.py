import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

###############################################
# Low pT and detached tracks from pixel quadruplets
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS
detachedQuadStepClusters = _cfg.clusterRemoverForIter("DetachedQuadStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(detachedQuadStepClusters, _cfg.clusterRemoverForIter("DetachedQuadStep", _eraName, _postfix))

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
import RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff
detachedQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix = dict(skipClusters = cms.InputTag('detachedQuadStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('detachedQuadStepClusters'))
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(detachedQuadStepSeedLayers,
    layerList = RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff.PixelSeedMergerQuadruplets.layerList.value()
)
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1QuadProp.toModify(detachedQuadStepSeedLayers,
    layerList = RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff.PixelSeedMergerQuadruplets.layerList.value()
)
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(detachedQuadStepSeedLayers, 
    layerList = RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff.PixelSeedMergerQuadruplets.layerList.value()
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpotFixedZ_cfi import globalTrackingRegionFromBeamSpotFixedZ as _globalTrackingRegionFromBeamSpotFixedZ
detachedQuadStepTrackingRegions = _globalTrackingRegionFromBeamSpotFixedZ.clone(RegionPSet = dict(
    ptMin = 0.3,
    originHalfLength = 15.0,
    originRadius = 1.5
))
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toReplaceWith(detachedQuadStepTrackingRegions, _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 0.3,
    originRadius = 0.5,
    nSigmaZ = 4.0
)))
trackingPhase2PU140.toReplaceWith(detachedQuadStepTrackingRegions, _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 0.45,
    originRadius = 0.7,
    nSigmaZ = 4.0
)))

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
detachedQuadStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "detachedQuadStepSeedLayers",
    trackingRegions = "detachedQuadStepTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletLargeTipEDProducer_cfi import pixelTripletLargeTipEDProducer as _pixelTripletLargeTipEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
detachedQuadStepHitTriplets = _pixelTripletLargeTipEDProducer.clone(
    doublets = "detachedQuadStepHitDoublets",
    produceIntermediateHitTriplets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelQuadrupletEDProducer_cfi import pixelQuadrupletEDProducer as _pixelQuadrupletEDProducer
detachedQuadStepHitQuadruplets = _pixelQuadrupletEDProducer.clone(
    triplets = "detachedQuadStepHitTriplets",
    extraHitRZtolerance = detachedQuadStepHitTriplets.extraHitRZtolerance,
    extraHitRPhitolerance = detachedQuadStepHitTriplets.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 500, value2 = 100,
        enabled = True,
    ),
    extraPhiTolerance = dict(
        pt1    = 0.4, pt2    = 1,
        value1 = 0.2, value2 = 0.05,
        enabled = True,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
)
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
trackingPhase1.toModify(detachedQuadStepHitDoublets, layerPairs = [0,1,2]) # layer pairs (0,1), (1,2), (2,3)
trackingPhase1.toReplaceWith(detachedQuadStepHitQuadruplets, _caHitQuadrupletEDProducer.clone(
    doublets = "detachedQuadStepHitDoublets",
    extraHitRPhitolerance = detachedQuadStepHitTriplets.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 500, value2 = 100,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0011,
    CAPhiCut = 0,
))

from RecoPixelVertexing.PixelTriplets.pixelQuadrupletMergerEDProducer_cfi import pixelQuadrupletMergerEDProducer as _pixelQuadrupletMergerEDProducer
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *
_detachedQuadStepHitQuadrupletsMerging = _pixelQuadrupletMergerEDProducer.clone(
    triplets = "detachedQuadStepHitTriplets",
    layerList = dict(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
detachedQuadStepSeeds = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets = "detachedQuadStepHitQuadruplets",
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    ),
)
# temporary...
_detachedQuadStepHitQuadrupletsMerging.SeedCreatorPSet = cms.PSet(
    ComponentName = cms.string("SeedFromConsecutiveHitsTripletOnlyCreator"),
    MinOneOverPtError = detachedQuadStepSeeds.MinOneOverPtError,
    OriginTransverseErrorMultiplier = detachedQuadStepSeeds.OriginTransverseErrorMultiplier,
    SeedMomentumForBOFF = detachedQuadStepSeeds.SeedMomentumForBOFF,
    TTRHBuilder = detachedQuadStepSeeds.TTRHBuilder,
    forceKinematicWithRegionDirection = detachedQuadStepSeeds.forceKinematicWithRegionDirection,
    magneticField = detachedQuadStepSeeds.magneticField,
    propagator = detachedQuadStepSeeds.propagator,
)
_detachedQuadStepHitQuadrupletsMerging.SeedComparitorPSet = detachedQuadStepSeeds.SeedComparitorPSet

trackingPhase1PU70.toModify(detachedQuadStepHitTriplets, produceIntermediateHitTriplets=False, produceSeedingHitSets=True)
trackingPhase1PU70.toReplaceWith(detachedQuadStepHitQuadruplets, _detachedQuadStepHitQuadrupletsMerging)
trackingPhase1PU70.toModify(detachedQuadStepSeeds, SeedComparitorPSet=cms.PSet(ComponentName=cms.string("none")))


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_detachedQuadStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
)
detachedQuadStepTrajectoryFilterBase = _detachedQuadStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 0,
    minGoodStripCharge = dict(refToPSet_ = 'SiStripClusterChargeCutLoose')
)
trackingPhase1PU70.toReplaceWith(detachedQuadStepTrajectoryFilterBase,
    _detachedQuadStepTrajectoryFilterBase.clone(
        maxLostHitsFraction = 1./10.,
        constantValueForLostHitsFractionFilter = 0.501,
    )
)
trackingPhase2PU140.toReplaceWith(detachedQuadStepTrajectoryFilterBase,
    _detachedQuadStepTrajectoryFilterBase.clone(
        maxLostHitsFraction = 1./10.,
        constantValueForLostHitsFractionFilter = 0.301,
    )
)
detachedQuadStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('detachedQuadStepTrajectoryFilterBase'))]
)
trackingPhase1PU70.toModify(detachedQuadStepTrajectoryFilter,
    filters = detachedQuadStepTrajectoryFilter.filters.value()+[cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)
trackingPhase2PU140.toModify(detachedQuadStepTrajectoryFilter,
    filters = detachedQuadStepTrajectoryFilter.filters.value()+[cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)


import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
detachedQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'detachedQuadStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 9.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight'),
)
trackingPhase1PU70.toModify(detachedQuadStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutNone")
)
trackingPhase2PU140.toModify(detachedQuadStepChi2Est,
    MaxChi2 = 16.0,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutNone")
)


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
detachedQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = dict(refToPSet_ = 'detachedQuadStepTrajectoryFilter'),
    maxCand = 3,
    alwaysUseInvalidHits = True,
    estimator = 'detachedQuadStepChi2Est',
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
)
trackingPhase1PU70.toModify(detachedQuadStepTrajectoryBuilder,
    maxCand = 2,
    alwaysUseInvalidHits = False,
)
trackingPhase2PU140.toModify(detachedQuadStepTrajectoryBuilder,
    maxCand = 2,
    alwaysUseInvalidHits = False,
)

# MAKING OF TRACK CANDIDATES
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
detachedQuadStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = cms.string('detachedQuadStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.13),
    allowSharedFirstHit = cms.bool(True)
)
trackingPhase1PU70.toModify(detachedQuadStepTrajectoryCleanerBySharedHits,
    fractionShared = 0.095
)
trackingPhase2PU140.toModify(detachedQuadStepTrajectoryCleanerBySharedHits,
    fractionShared = 0.09
)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
detachedQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'detachedQuadStepSeeds',
    clustersToSkip = cms.InputTag('detachedQuadStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = dict(refToPSet_ = 'detachedQuadStepTrajectoryBuilder'),
    TrajectoryCleaner = 'detachedQuadStepTrajectoryCleanerBySharedHits',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
trackingPhase2PU140.toModify(detachedQuadStepTrackCandidates,
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag("detachedQuadStepClusters")
)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
detachedQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = 'detachedQuadStep',
    src = 'detachedQuadStepTrackCandidates',
    Fitter = 'FlexibleKFFittingSmoother',
)

# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
detachedQuadStepClassifier1 = TrackMVAClassifierDetached.clone(
    src = 'detachedQuadStepTracks',
    GBRForestLabel = 'MVASelectorIter3_13TeV',
    qualityCuts = [-0.5,0.0,0.5]
)
detachedQuadStepClassifier2 = TrackMVAClassifierPrompt.clone(
    src = 'detachedQuadStepTracks',
    GBRForestLabel = 'MVASelectorIter0_13TeV',
    qualityCuts = [-0.2,0.0,0.4]
)

# For Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedQuadStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'detachedQuadStepTracks',
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedQuadStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedQuadStepTrkLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.3, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedQuadStepVtxTight',
            preFilterName = 'detachedQuadStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 0.9, 3.0 ),
            dz_par2 = ( 0.9, 3.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedQuadStepTrkTight',
            preFilterName = 'detachedQuadStepTrkLoose',
            chi2n_par = 0.35,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 4,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedQuadStepVtx',
            preFilterName = 'detachedQuadStepVtxTight',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 3.0 ),
            dz_par1 = ( 0.8, 3.0 ),
            d0_par2 = ( 0.8, 3.0 ),
            dz_par2 = ( 0.8, 3.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedQuadStepTrk',
            preFilterName = 'detachedQuadStepTrkTight',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 4,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.8, 4.0 ),
            dz_par2 = ( 0.8, 4.0 )
        )
    ]
) #end of clone

trackingPhase2PU140.toModify(detachedQuadStepSelector,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedQuadStepVtxLoose',
            chi2n_par = 1.0,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedQuadStepTrkLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.3, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedQuadStepVtxTight',
            preFilterName = 'detachedQuadStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 0.9, 3.0 ),
            dz_par2 = ( 0.9, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedQuadStepTrkTight',
            preFilterName = 'detachedQuadStepTrkLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedQuadStepVtx',
            preFilterName = 'detachedQuadStepVtxTight',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 3.0 ),
            dz_par1 = ( 0.8, 3.0 ),
            d0_par2 = ( 0.8, 3.0 ),
            dz_par2 = ( 0.8, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedQuadStepTrk',
            preFilterName = 'detachedQuadStepTrkTight',
            chi2n_par = 0.45,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.8, 4.0 ),
            dz_par2 = ( 0.8, 4.0 )
            )
        ), #end of vpset
     vertices = "pixelVertices"
    ) #end of clone

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
detachedQuadStep = ClassifierMerger.clone()
detachedQuadStep.inputClassifiers=['detachedQuadStepClassifier1','detachedQuadStepClassifier2']

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
trackingPhase1PU70.toReplaceWith(detachedQuadStep, RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = [
        'detachedQuadStepTracks',
        'detachedQuadStepTracks',
    ],
    hasSelector = [1,1],
    shareFrac = cms.double(0.095),
    indivShareFrac = [0.095, 0.095],
    selectedTrackQuals = [
        cms.InputTag("detachedQuadStepSelector","detachedQuadStepVtx"),
        cms.InputTag("detachedQuadStepSelector","detachedQuadStepTrk")
    ],
    setsToMerge = [cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )],
    writeOnlyTrkQuals = True
))

trackingPhase2PU140.toReplaceWith(detachedQuadStep, RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('detachedQuadStepTracks'),
                                   cms.InputTag('detachedQuadStepTracks')),
    hasSelector=cms.vint32(1,1),
    shareFrac = cms.double(0.09),
    indivShareFrac=cms.vdouble(0.09,0.09),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("detachedQuadStepSelector","detachedQuadStepVtx"),
                                       cms.InputTag("detachedQuadStepSelector","detachedQuadStepTrk")),
    setsToMerge = cms.VPSet(cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
    )
)

DetachedQuadStep = cms.Sequence(detachedQuadStepClusters*
                                detachedQuadStepSeedLayers*
                                detachedQuadStepTrackingRegions*
                                detachedQuadStepHitDoublets*
                                detachedQuadStepHitTriplets*
                                detachedQuadStepHitQuadruplets*
                                detachedQuadStepSeeds*
                                detachedQuadStepTrackCandidates*
                                detachedQuadStepTracks*
                                detachedQuadStepClassifier1*detachedQuadStepClassifier2*
                                detachedQuadStep)
trackingPhase1.toReplaceWith(DetachedQuadStep, DetachedQuadStep.copyAndExclude([detachedQuadStepHitTriplets]))
_DetachedQuadStep_Phase1PU70 = DetachedQuadStep.copyAndExclude([detachedQuadStepClassifier1])
_DetachedQuadStep_Phase1PU70.replace(detachedQuadStepClassifier2, detachedQuadStepSelector)
trackingPhase1PU70.toReplaceWith(DetachedQuadStep, _DetachedQuadStep_Phase1PU70)
trackingPhase2PU140.toReplaceWith(DetachedQuadStep, _DetachedQuadStep_Phase1PU70)
