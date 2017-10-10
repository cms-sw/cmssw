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
import RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi
detachedQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi.PixelLayerQuadruplets.clone(
    BPix = dict(skipClusters = cms.InputTag('detachedQuadStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('detachedQuadStepClusters'))
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpotFixedZ_cfi import globalTrackingRegionFromBeamSpotFixedZ as _globalTrackingRegionFromBeamSpotFixedZ
detachedQuadStepTrackingRegions = _globalTrackingRegionFromBeamSpotFixedZ.clone(RegionPSet = dict(
    ptMin = 0.3,
    originHalfLength = 15.0,
    originRadius = 1.5
))
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toReplaceWith(detachedQuadStepTrackingRegions, _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 0.45,
    originRadius = 0.9,
    nSigmaZ = 5.0
)))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
pp_on_XeXe_2017.toReplaceWith(detachedQuadStepTrackingRegions, 
                              _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
            fixedError = 3.75,
            ptMin = 0.8,
            originRadius = 1.5
            )
                                                                      )
)

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
detachedQuadStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "detachedQuadStepSeedLayers",
    trackingRegions = "detachedQuadStepTrackingRegions",
    layerPairs = [0,1,2], # layer pairs (0,1), (1,2), (2,3),
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletLargeTipEDProducer_cfi import pixelTripletLargeTipEDProducer as _pixelTripletLargeTipEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
detachedQuadStepHitQuadruplets = _caHitQuadrupletEDProducer.clone(
    doublets = "detachedQuadStepHitDoublets",
    extraHitRPhitolerance = _pixelTripletLargeTipEDProducer.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 500, value2 = 100,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0011,
    CAPhiCut = 0,
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


from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1QuadProp.toModify(detachedQuadStepHitDoublets, layerPairs = [0])
detachedQuadStepHitTriplets = _pixelTripletLargeTipEDProducer.clone(
    doublets = "detachedQuadStepHitDoublets",
    produceIntermediateHitTriplets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelQuadrupletEDProducer_cfi import pixelQuadrupletEDProducer as _pixelQuadrupletEDProducer
_detachedQuadStepHitQuadruplets_propagation = _pixelQuadrupletEDProducer.clone(
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
trackingPhase1QuadProp.toReplaceWith(detachedQuadStepHitQuadruplets, _detachedQuadStepHitQuadruplets_propagation)


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
trackingPhase2PU140.toReplaceWith(detachedQuadStepTrajectoryFilterBase,
    _detachedQuadStepTrajectoryFilterBase.clone(
        maxLostHitsFraction = 1./10.,
        constantValueForLostHitsFractionFilter = 0.301,
    )
)
detachedQuadStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('detachedQuadStepTrajectoryFilterBase'))]
)
trackingPhase2PU140.toModify(detachedQuadStepTrajectoryFilter,
    filters = detachedQuadStepTrajectoryFilter.filters.value()+[cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

pp_on_XeXe_2017.toModify(detachedQuadStepTrajectoryFilterBase, minPt=0.9)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
detachedQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'detachedQuadStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 9.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight'),
)
trackingPhase2PU140.toModify(detachedQuadStepChi2Est,
    MaxChi2 = 12.0,
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
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
detachedQuadStep = TrackMVAClassifierDetached.clone(
    src = 'detachedQuadStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorDetachedQuadStep_Phase1'),
    qualityCuts = [-0.5,0.0,0.5],
)


# For Phase2PU140
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedQuadStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'detachedQuadStepTracks',
    trackSelectors = [
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
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
            )
    ] #end of vpset
) #end of clone


from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
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
                                detachedQuadStepHitQuadruplets*
                                detachedQuadStepSeeds*
                                detachedQuadStepTrackCandidates*
                                detachedQuadStepTracks*
                                detachedQuadStep)
_DetachedQuadStep_Phase1Prop = DetachedQuadStep.copy()
_DetachedQuadStep_Phase1Prop.replace(detachedQuadStepHitDoublets, detachedQuadStepHitDoublets+detachedQuadStepHitTriplets)
trackingPhase1QuadProp.toReplaceWith(DetachedQuadStep, _DetachedQuadStep_Phase1Prop)
_DetachedQuadStep_Phase2PU140 = DetachedQuadStep.copy()
_DetachedQuadStep_Phase2PU140.replace(detachedQuadStep, detachedQuadStepSelector+detachedQuadStep)
trackingPhase2PU140.toReplaceWith(DetachedQuadStep, _DetachedQuadStep_Phase2PU140)
