import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

# NEW CLUSTERS (remove previously used clusters)
lowPtQuadStepClusters = _cfg.clusterRemoverForIter("LowPtQuadStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(lowPtQuadStepClusters, _cfg.clusterRemoverForIter("LowPtQuadStep", _eraName, _postfix))


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi
lowPtQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi.PixelLayerQuadruplets.clone(
    BPix = dict(skipClusters = cms.InputTag('lowPtQuadStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('lowPtQuadStepClusters'))
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
lowPtQuadStepTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 0.15,
    originRadius = 0.02,
    nSigmaZ = 4.0
))
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase1QuadProp.toModify(lowPtQuadStepTrackingRegions, RegionPSet = dict(ptMin = 0.2))
trackingPhase2PU140.toModify(lowPtQuadStepTrackingRegions, RegionPSet = dict(ptMin = 0.35,originRadius = 0.025))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
pp_on_XeXe_2017.toReplaceWith(lowPtQuadStepTrackingRegions, 
                              _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
            fixedError = 0.5,
            ptMin = 0.25,
            originRadius = 0.02
            )
                                                                      )
)

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
lowPtQuadStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "lowPtQuadStepSeedLayers",
    trackingRegions = "lowPtQuadStepTrackingRegions",
    layerPairs = [0,1,2], # layer pairs (0,1), (1,2), (2,3)
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtQuadStepHitQuadruplets = _caHitQuadrupletEDProducer.clone(
    doublets = "lowPtQuadStepHitDoublets",
    extraHitRPhitolerance = _pixelTripletHLTEDProducer.extraHitRPhitolerance,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(),
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 1000, value2 = 150,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0017,
    CAPhiCut = 0.3,
)
trackingPhase2PU140.toModify(lowPtQuadStepHitQuadruplets,CAThetaCut = 0.0015,CAPhiCut = 0.25)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
lowPtQuadStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "lowPtQuadStepHitQuadruplets",
)

trackingPhase1QuadProp.toModify(lowPtQuadStepHitDoublets, layerPairs = [0])
lowPtQuadStepHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "lowPtQuadStepHitDoublets",
    produceIntermediateHitTriplets = True,
    SeedComparitorPSet = lowPtQuadStepHitQuadruplets.SeedComparitorPSet,
)
from RecoPixelVertexing.PixelTriplets.pixelQuadrupletEDProducer_cfi import pixelQuadrupletEDProducer as _pixelQuadrupletEDProducer
_lowPtQuadStepHitQuadruplets_propagation = _pixelQuadrupletEDProducer.clone(
    triplets = "lowPtQuadStepHitTriplets",
    extraHitRZtolerance = lowPtQuadStepHitTriplets.extraHitRZtolerance,
    extraHitRPhitolerance = lowPtQuadStepHitTriplets.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8  , pt2    = 2,
        value1 = 2000, value2 = 100,
        enabled = True,
    ),
    extraPhiTolerance = dict(
        pt1    = 0.3, pt2    = 1,
        value1 = 0.4, value2 = 0.05,
        enabled = True,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    SeedComparitorPSet = lowPtQuadStepHitTriplets.SeedComparitorPSet,
)
trackingPhase1QuadProp.toReplaceWith(lowPtQuadStepHitQuadruplets, _lowPtQuadStepHitQuadruplets_propagation)



# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_lowPtQuadStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
)
lowPtQuadStepTrajectoryFilterBase = _lowPtQuadStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 0,
    minGoodStripCharge = dict(refToPSet_ = 'SiStripClusterChargeCutLoose')
)
trackingPhase2PU140.toReplaceWith(lowPtQuadStepTrajectoryFilterBase, _lowPtQuadStepTrajectoryFilterBase)

pp_on_XeXe_2017.toModify(lowPtQuadStepTrajectoryFilterBase, minPt=0.3)

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtQuadStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('lowPtQuadStepTrajectoryFilterBase'))]
)
trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryFilter,
    filters = lowPtQuadStepTrajectoryFilter.filters.value() + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'lowPtQuadStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 9.0,
    clusterChargeCut = dict(refToPSet_ = ('SiStripClusterChargeCutTight')),
)
trackingPhase2PU140.toModify(lowPtQuadStepChi2Est,
    MaxChi2 = 16.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = dict(refToPSet_ = 'lowPtQuadStepTrajectoryFilter'),
    maxCand = 4,
    estimator = cms.string('lowPtQuadStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
)
trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryBuilder,
    minNrOfHitsForRebuild = 1,
    keepOriginalIfRebuildFails = True,
)


# MAKING OF TRACK CANDIDATES
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits as _trajectoryCleanerBySharedHits
lowPtQuadStepTrajectoryCleanerBySharedHits = _trajectoryCleanerBySharedHits.clone(
    ComponentName = 'lowPtQuadStepTrajectoryCleanerBySharedHits',
    fractionShared = 0.16,
    allowSharedFirstHit = True
)
trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'lowPtQuadStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = dict(refToPSet_ = 'lowPtQuadStepTrajectoryBuilder'),
    TrajectoryCleaner = 'lowPtQuadStepTrajectoryCleanerBySharedHits',
    clustersToSkip = cms.InputTag('lowPtQuadStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
trackingPhase2PU140.toModify(lowPtQuadStepTrackCandidates,
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag("lowPtQuadStepClusters")
)
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtQuadStepTrackCandidates',
    AlgorithmName = 'lowPtQuadStep',
    Fitter = 'FlexibleKFFittingSmoother',
)



# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
lowPtQuadStep =  TrackMVAClassifierPrompt.clone(
    src = 'lowPtQuadStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorLowPtQuadStep_Phase1'),
    qualityCuts = [-0.7,-0.35,-0.15],
)

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
LowPtQuadStep = cms.Sequence(lowPtQuadStepClusters*
                             lowPtQuadStepSeedLayers*
                             lowPtQuadStepTrackingRegions*
                             lowPtQuadStepHitDoublets*
                             lowPtQuadStepHitQuadruplets*
                             lowPtQuadStepSeeds*
                             lowPtQuadStepTrackCandidates*
                             lowPtQuadStepTracks*
                             lowPtQuadStep)
_LowPtQuadStep_Phase1Prop = LowPtQuadStep.copy()
_LowPtQuadStep_Phase1Prop.replace(lowPtQuadStepHitDoublets, lowPtQuadStepHitDoublets+lowPtQuadStepHitTriplets)
trackingPhase1QuadProp.toReplaceWith(LowPtQuadStep, _LowPtQuadStep_Phase1Prop)
_LowPtQuadStep_Phase2PU140 = LowPtQuadStep.copy()
_LowPtQuadStep_Phase2PU140.replace(lowPtQuadStep, lowPtQuadStepSelector)
trackingPhase2PU140.toReplaceWith(LowPtQuadStep, _LowPtQuadStep_Phase2PU140)
