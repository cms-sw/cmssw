import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

from RecoTracker.TkSeedGenerator.trackerClusterCheck_cfi import trackerClusterCheck as _trackerClusterCheck
trackerClusterCheckPreSplitting = _trackerClusterCheck.clone(
    PixelClusterCollectionLabel = 'siPixelClustersPreSplitting'
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
import RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff
initialStepSeedLayersPreSplitting = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
initialStepSeedLayersPreSplitting.FPix.HitProducer = 'siPixelRecHitsPreSplitting'
initialStepSeedLayersPreSplitting.BPix.HitProducer = 'siPixelRecHitsPreSplitting'
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(initialStepSeedLayersPreSplitting,
    layerList = RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff.PixelSeedMergerQuadruplets.layerList.value()
)
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1QuadProp.toModify(initialStepSeedLayersPreSplitting,
    layerList = RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff.PixelSeedMergerQuadruplets.layerList.value()
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
initialStepTrackingRegionsPreSplitting = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 0.6,
    originRadius = 0.02,
    nSigmaZ = 4.0
))
trackingPhase1.toModify(initialStepTrackingRegionsPreSplitting, RegionPSet = dict(ptMin = 0.5))

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
initialStepHitDoubletsPreSplitting = _hitPairEDProducer.clone(
    seedingLayers = "initialStepSeedLayersPreSplitting",
    trackingRegions = "initialStepTrackingRegionsPreSplitting",
    clusterCheck = "trackerClusterCheckPreSplitting",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
initialStepHitTripletsPreSplitting = _pixelTripletHLTEDProducer.clone(
    doublets = "initialStepHitDoubletsPreSplitting",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(
        clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting'
    ),
)
from RecoPixelVertexing.PixelTriplets.pixelQuadrupletEDProducer_cfi import pixelQuadrupletEDProducer as _pixelQuadrupletEDProducer
trackingPhase1.toModify(initialStepHitDoubletsPreSplitting, layerPairs = [0,1,2]) # layer pairs (0,1), (1,2), (2,3)
initialStepHitQuadrupletsPreSplitting = _pixelQuadrupletEDProducer.clone(
    triplets = "initialStepHitTripletsPreSplitting",
    extraHitRZtolerance = initialStepHitTripletsPreSplitting.extraHitRZtolerance,
    extraHitRPhitolerance = initialStepHitTripletsPreSplitting.extraHitRPhitolerance,
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
    SeedComparitorPSet = initialStepHitTripletsPreSplitting.SeedComparitorPSet
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
initialStepSeedsPreSplitting = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "initialStepHitTripletsPreSplitting",
)

from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
trackingPhase1.toReplaceWith(initialStepHitQuadrupletsPreSplitting, _caHitQuadrupletEDProducer.clone(
    doublets = "initialStepHitDoubletsPreSplitting",
    extraHitRPhitolerance = initialStepHitTripletsPreSplitting.extraHitRPhitolerance,
    SeedComparitorPSet = initialStepHitTripletsPreSplitting.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 200, value2 = 50,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0012,
    CAPhiCut = 0.2,
))

trackingPhase1QuadProp.toModify(initialStepHitTripletsPreSplitting,
    produceSeedingHitSets = False,
    produceIntermediateHitTriplets = True,
)
trackingPhase1.toModify(initialStepSeedsPreSplitting, seedingHitSets = "initialStepHitQuadrupletsPreSplitting")
trackingPhase1QuadProp.toModify(initialStepSeedsPreSplitting, seedingHitSets = "initialStepHitQuadrupletsPreSplitting")


# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
initialStepTrajectoryFilterBasePreSplitting = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2,
    maxCCCLostHits = 0,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    )
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(initialStepTrajectoryFilterBasePreSplitting, maxCCCLostHits = 2)
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
initialStepTrajectoryFilterShapePreSplitting = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
initialStepTrajectoryFilterPreSplitting = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterBasePreSplitting')),
        cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterShapePreSplitting'))),
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
initialStepChi2EstPreSplitting = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('initialStepChi2EstPreSplitting'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose')),
    pTChargeCutThreshold = cms.double(15.)
)
_tracker_apv_vfp30_2016.toModify(initialStepChi2EstPreSplitting,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutTiny")
)

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
initialStepTrajectoryBuilderPreSplitting = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('initialStepTrajectoryFilterPreSplitting')),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = cms.string('initialStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
initialStepTrackCandidatesPreSplitting = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('initialStepSeedsPreSplitting'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('initialStepTrajectoryBuilderPreSplitting')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )
initialStepTrackCandidatesPreSplitting.MeasurementTrackerEvent = 'MeasurementTrackerEventPreSplitting'

# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
initialStepTracksPreSplitting = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'initialStepTrackCandidatesPreSplitting',
    AlgorithmName = cms.string('initialStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )
initialStepTracksPreSplitting.MeasurementTrackerEvent = 'MeasurementTrackerEventPreSplitting'

#vertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
firstStepPrimaryVerticesPreSplitting = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
firstStepPrimaryVerticesPreSplitting.TrackLabel = cms.InputTag("initialStepTracksPreSplitting")
firstStepPrimaryVerticesPreSplitting.vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               )
      ]
    )

#Jet Core emulation to identify jet-tracks
from RecoTracker.IterativeTracking.JetCoreRegionalStep_cff import initialStepTrackRefsForJets, caloTowerForTrk, ak4CaloJetsForTrk, jetsForCoreTracking
initialStepTrackRefsForJetsPreSplitting = initialStepTrackRefsForJets.clone(
    src = 'initialStepTracksPreSplitting')
caloTowerForTrkPreSplitting = caloTowerForTrk.clone()
ak4CaloJetsForTrkPreSplitting = ak4CaloJetsForTrk.clone(
    src = 'caloTowerForTrkPreSplitting',
    srcPVs = 'firstStepPrimaryVerticesPreSplitting')
jetsForCoreTrackingPreSplitting = jetsForCoreTracking.clone(
    src = 'ak4CaloJetsForTrkPreSplitting')

#Cluster Splitting
from RecoLocalTracker.SubCollectionProducers.jetCoreClusterSplitter_cfi import jetCoreClusterSplitter
siPixelClusters = jetCoreClusterSplitter.clone(
    pixelClusters = cms.InputTag('siPixelClustersPreSplitting'),
    vertices      = 'firstStepPrimaryVerticesPreSplitting',
    cores         = 'jetsForCoreTrackingPreSplitting'
)

# Final sequence
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import siPixelRecHits
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import MeasurementTrackerEvent
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
InitialStepPreSplitting = cms.Sequence(trackerClusterCheckPreSplitting*
                                       initialStepSeedLayersPreSplitting*
                                       initialStepTrackingRegionsPreSplitting*
                                       initialStepHitDoubletsPreSplitting*
                                       initialStepHitTripletsPreSplitting*
                                       initialStepSeedsPreSplitting*
                                       initialStepTrackCandidatesPreSplitting*
                                       initialStepTracksPreSplitting*
                                       firstStepPrimaryVerticesPreSplitting*
                                       initialStepTrackRefsForJetsPreSplitting*
                                       caloTowerForTrkPreSplitting*
                                       ak4CaloJetsForTrkPreSplitting*
                                       jetsForCoreTrackingPreSplitting*
                                       siPixelClusters*
                                       siPixelRecHits*
                                       MeasurementTrackerEvent*
                                       siPixelClusterShapeCache)

_InitialStepPreSplitting_trackingPhase1 = InitialStepPreSplitting.copy()
_InitialStepPreSplitting_trackingPhase1.replace(initialStepHitTripletsPreSplitting, initialStepHitTripletsPreSplitting*initialStepHitQuadrupletsPreSplitting)
trackingPhase1.toReplaceWith(InitialStepPreSplitting, _InitialStepPreSplitting_trackingPhase1.copyAndExclude([initialStepHitTripletsPreSplitting]))
trackingPhase1QuadProp.toReplaceWith(InitialStepPreSplitting, _InitialStepPreSplitting_trackingPhase1)

# Although InitialStepPreSplitting is not really part of LowPU/Run1/Phase1PU70
# tracking, we use it to get siPixelClusters and siPixelRecHits
# collections for non-splitted pixel clusters. All modules before
# iterTracking sequence use siPixelClustersPreSplitting and
# siPixelRecHitsPreSplitting for that purpose.
#
# If siPixelClusters would be defined in
# RecoLocalTracker.Configuration.RecoLocalTracker_cff, we would have a
# situation where
# - LowPU/Phase1PU70 has siPixelClusters defined in RecoLocalTracker_cff
# - everything else has siPixelClusters defined here
# and this leads to a mess. The way it is done here we have only
# one place (within Reconstruction_cff) where siPixelClusters
# module is defined.
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(siPixelClusters, _siPixelClusters)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toReplaceWith(siPixelClusters, _siPixelClusters)
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toReplaceWith(siPixelClusters, _siPixelClusters)
_InitialStepPreSplitting_LowPU_Phase1PU70 = cms.Sequence(
    siPixelClusters +
    siPixelRecHits +
    MeasurementTrackerEvent +
    siPixelClusterShapeCache
)
trackingLowPU.toReplaceWith(InitialStepPreSplitting, _InitialStepPreSplitting_LowPU_Phase1PU70)
trackingPhase1PU70.toReplaceWith(InitialStepPreSplitting, _InitialStepPreSplitting_LowPU_Phase1PU70)
trackingPhase2PU140.toReplaceWith(InitialStepPreSplitting, _InitialStepPreSplitting_LowPU_Phase1PU70)
