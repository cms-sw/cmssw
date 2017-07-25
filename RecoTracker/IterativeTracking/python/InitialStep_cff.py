import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
import RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff
initialStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(initialStepSeedLayers,
    layerList = RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff.PixelSeedMergerQuadruplets.layerList.value()
)
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1QuadProp.toModify(initialStepSeedLayers,
    layerList = [
        'BPix1+BPix2+BPix3',
        'BPix1+BPix2+FPix1_pos',
        'BPix1+BPix2+FPix1_neg',
        'BPix1+FPix1_pos+FPix2_pos',
        'BPix1+FPix1_neg+FPix2_neg'
    ]
)
trackingPhase2PU140.toModify(initialStepSeedLayers,
    layerList = RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff.PixelSeedMergerQuadruplets.layerList.value()
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
initialStepTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 0.6,
    originRadius = 0.02,
    nSigmaZ = 4.0
))
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase1.toModify(initialStepTrackingRegions, RegionPSet = dict(ptMin = 0.5))
trackingPhase2PU140.toModify(initialStepTrackingRegions, RegionPSet = dict(ptMin = 0.6,originRadius = 0.03))

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
initialStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "initialStepSeedLayers",
    trackingRegions = "initialStepTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
initialStepHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "initialStepHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
initialStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "initialStepHitTriplets",
)
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
_initialStepCAHitQuadruplets = _caHitQuadrupletEDProducer.clone(
    doublets = "initialStepHitDoublets",
    extraHitRPhitolerance = initialStepHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = initialStepHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 200, value2 = 50,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0012,
    CAPhiCut = 0.2,
)
initialStepHitQuadruplets = _initialStepCAHitQuadruplets.clone()
trackingPhase1.toModify(initialStepHitDoublets, layerPairs = [0,1,2]) # layer pairs (0,1), (1,2), (2,3)

from RecoPixelVertexing.PixelTriplets.pixelQuadrupletEDProducer_cfi import pixelQuadrupletEDProducer as _pixelQuadrupletEDProducer
trackingPhase1QuadProp.toReplaceWith(initialStepHitQuadruplets, _pixelQuadrupletEDProducer.clone(
    triplets = "initialStepHitTriplets",
    extraHitRZtolerance = initialStepHitTriplets.extraHitRZtolerance,
    extraHitRPhitolerance = initialStepHitTriplets.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 200, value2 = 100,
        enabled = True,
    ),
    extraPhiTolerance = dict(
        pt1    = 0.6, pt2    = 1,
        value1 = 0.15, value2 = 0.1,
        enabled = True,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    SeedComparitorPSet = initialStepHitTriplets.SeedComparitorPSet
))

trackingPhase2PU140.toModify(initialStepHitDoublets, layerPairs = [0,1,2]) # layer pairs (0,1), (1,2), (2,3)
trackingPhase2PU140.toModify(initialStepHitQuadruplets,
    CAThetaCut = 0.0010,
    CAPhiCut = 0.175,
)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
_initialStepSeedsConsecutiveHitsTripletOnly = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets = "initialStepHitTriplets",
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    ),
)
trackingPhase1QuadProp.toReplaceWith(initialStepSeeds, _initialStepSeedsConsecutiveHitsTripletOnly)
trackingPhase1.toReplaceWith(initialStepSeeds, _initialStepSeedsConsecutiveHitsTripletOnly.clone(
        seedingHitSets = "initialStepHitQuadruplets"
))
trackingPhase2PU140.toReplaceWith(initialStepSeeds, _initialStepSeedsConsecutiveHitsTripletOnly.clone(
        seedingHitSets = "initialStepHitQuadruplets"
))


# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_initialStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2,
)
initialStepTrajectoryFilterBase = _initialStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 0,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(initialStepTrajectoryFilterBase, maxCCCLostHits = 2)
initialStepTrajectoryFilterInOut = initialStepTrajectoryFilterBase.clone(
    minimumNumberOfHits = 4,
    seedExtension = 1,
    strictSeedExtension = True, # don't allow inactive
    pixelSeedExtension = True,
)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(initialStepTrajectoryFilterBase, _initialStepTrajectoryFilterBase)
trackingPhase2PU140.toReplaceWith(initialStepTrajectoryFilterBase, _initialStepTrajectoryFilterBase)

import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
initialStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
initialStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterBase')),
    #    cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterShape'))
    ),
)

trackingPhase2PU140.toReplaceWith(initialStepTrajectoryFilter, TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2
    )
)
import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
initialStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('initialStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose')),
    pTChargeCutThreshold = cms.double(15.)
)
_tracker_apv_vfp30_2016.toModify(initialStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutTiny")
)
trackingPhase2PU140.toModify(initialStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
)


import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
initialStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('initialStepTrajectoryFilter')),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = cms.string('initialStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )
trackingLowPU.toModify(initialStepTrajectoryBuilder, maxCand = 5)
trackingPhase1.toModify(initialStepTrajectoryBuilder,
    minNrOfHitsForRebuild = 1,
    keepOriginalIfRebuildFails = True,
)
trackingPhase1QuadProp.toModify(initialStepTrajectoryBuilder,
    minNrOfHitsForRebuild = 1,
    keepOriginalIfRebuildFails = True,
    inOutTrajectoryFilter = dict(refToPSet_ = "initialStepTrajectoryFilterInOut"),
    useSameTrajFilter = False
)

trackingPhase2PU140.toModify(initialStepTrajectoryBuilder,
    minNrOfHitsForRebuild = 1,
    keepOriginalIfRebuildFails = True,
)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
initialStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('initialStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('initialStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
initialStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'initialStepTrackCandidates',
    AlgorithmName = cms.string('initialStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )


#vertices
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices as _offlinePrimaryVertices
firstStepPrimaryVerticesUnsorted = _offlinePrimaryVertices.clone()
firstStepPrimaryVerticesUnsorted.TrackLabel = cms.InputTag("initialStepTracks")
firstStepPrimaryVerticesUnsorted.vertexCollections = [_offlinePrimaryVertices.vertexCollections[0].clone()]

from RecoJets.JetProducers.TracksForJets_cff import trackRefsForJets
initialStepTrackRefsForJets = trackRefsForJets.clone(src = cms.InputTag('initialStepTracks'))
from RecoJets.JetProducers.caloJetsForTrk_cff import *
from CommonTools.RecoAlgos.sortedPrimaryVertices_cfi import sortedPrimaryVertices as _sortedPrimaryVertices
firstStepPrimaryVertices = _sortedPrimaryVertices.clone(
    vertices = "firstStepPrimaryVerticesUnsorted",
    particles = "initialStepTrackRefsForJets",
)


# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *


initialStepClassifier1 = TrackMVAClassifierPrompt.clone()
initialStepClassifier1.src = 'initialStepTracks'
initialStepClassifier1.GBRForestLabel = 'MVASelectorIter0_13TeV'
initialStepClassifier1.qualityCuts = [-0.9,-0.8,-0.7]

from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepClassifier1
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStep
initialStepClassifier2 = detachedTripletStepClassifier1.clone()
initialStepClassifier2.src = 'initialStepTracks'
initialStepClassifier3 = lowPtTripletStep.clone()
initialStepClassifier3.src = 'initialStepTracks'

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
initialStep = ClassifierMerger.clone()
initialStep.inputClassifiers=['initialStepClassifier1','initialStepClassifier2','initialStepClassifier3']

trackingPhase1.toReplaceWith(initialStep, initialStepClassifier1.clone(
        GBRForestLabel = 'MVASelectorInitialStep_Phase1',
        qualityCuts = [-0.95,-0.85,-0.75],
))
trackingPhase1QuadProp.toReplaceWith(initialStep, initialStepClassifier1.clone(
        GBRForestLabel = 'MVASelectorInitialStep_Phase1',
        qualityCuts = [-0.95,-0.85,-0.75],
))

# For LowPU and Phase2PU140
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
initialStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'initialStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter0'),
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'initialStepLoose',
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'initialStepTight',
            preFilterName = 'initialStepLoose',
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'initialStepTight',
        ),
    ] #end of vpset
) #end of clone
trackingPhase2PU140.toModify(initialStepSelector,
    useAnyMVA = None,
    GBRForestLabel = None,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'initialStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.8, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'initialStepTight',
            preFilterName = 'initialStepLoose',
            chi2n_par = 1.4,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.7, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'initialStep',
            preFilterName = 'initialStepTight',
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.55, 4.0 )
            ),
        ), #end of vpset
) #end of clone



# Final sequence
InitialStep = cms.Sequence(initialStepSeedLayers*
                           initialStepTrackingRegions*
                           initialStepHitDoublets*
                           initialStepHitTriplets*
                           initialStepSeeds*
                           initialStepTrackCandidates*
                           initialStepTracks*
                           firstStepPrimaryVerticesUnsorted*
                           initialStepTrackRefsForJets*
                           caloJetsForTrk*
                           firstStepPrimaryVertices*
                           initialStepClassifier1*initialStepClassifier2*initialStepClassifier3*
                           initialStep)
_InitialStep_LowPU = InitialStep.copyAndExclude([firstStepPrimaryVerticesUnsorted, initialStepTrackRefsForJets, caloJetsForTrk, firstStepPrimaryVertices, initialStepClassifier1, initialStepClassifier2, initialStepClassifier3])
_InitialStep_LowPU.replace(initialStep, initialStepSelector)
trackingLowPU.toReplaceWith(InitialStep, _InitialStep_LowPU)
_InitialStep_Phase1QuadProp = InitialStep.copyAndExclude([initialStepClassifier2, initialStepClassifier3])
_InitialStep_Phase1QuadProp.replace(initialStepHitTriplets, initialStepHitTriplets+initialStepHitQuadruplets)
trackingPhase1QuadProp.toReplaceWith(InitialStep, _InitialStep_Phase1QuadProp)
_InitialStep_Phase1 = _InitialStep_Phase1QuadProp.copyAndExclude([initialStepHitTriplets])
trackingPhase1.toReplaceWith(InitialStep, _InitialStep_Phase1)
_InitialStep_trackingPhase2 = InitialStep.copyAndExclude([initialStepClassifier1, initialStepClassifier2, initialStepClassifier3])
_InitialStep_trackingPhase2.replace(initialStepHitTriplets, initialStepHitQuadruplets)
_InitialStep_trackingPhase2.replace(initialStep, initialStepSelector)
trackingPhase2PU140.toReplaceWith(InitialStep, _InitialStep_trackingPhase2)
