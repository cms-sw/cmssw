import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

###############################################
# Low pT and detached tracks from pixel quadruplets
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
_detachedQuadStepClustersBase = _trackClusterRemover.clone(
    maxChi2                                  = 9.0,
    trajectories                             = "highPtTripletStepTracks",
    pixelClusters                            = "siPixelClusters",
    stripClusters                            = "siStripClusters",
    oldClusterRemovalInfo                    = "highPtTripletStepClusters",
    TrackQuality                             = 'highPurity',
    minNumberOfLayersWithMeasBeforeFiltering = 0,
)
detachedQuadStepClusters = _detachedQuadStepClustersBase.clone(
    trackClassifier                          = "highPtTripletStep:QualityMasks",
)
eras.trackingPhase1PU70.toReplaceWith(detachedQuadStepClusters, _detachedQuadStepClustersBase.clone(
    trajectories                             = "lowPtTripletStepTracks",
    oldClusterRemovalInfo                    = "lowPtTripletStepClusters",
    overrideTrkQuals                         = "lowPtTripletStepSelector:lowPtTripletStep",
))


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
detachedQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix = dict(skipClusters = cms.InputTag('detachedQuadStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('detachedQuadStepClusters'))
)

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock as _RegionPsetFomBeamSpotBlock
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *
detachedQuadStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    OrderedHitsFactoryPSet = dict(
        SeedingLayers = 'detachedQuadStepSeedLayers',
        GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
    ),
    SeedCreatorPSet = dict(ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'),
    RegionFactoryPSet = dict(
        RegionPSet = dict(
            ptMin = 0.3,
            originHalfLength = 15.0,
            originRadius = 1.5
        )
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    ),
    SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    )
)
eras.trackingPhase1PU70.toModify(detachedQuadStepSeeds,
    RegionFactoryPSet = dict(
        RegionPSet = _RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            ptMin = 0.3,
            originRadius = 0.5,
            nSigmaZ = 4.0
        )
    )
)


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_detachedQuadStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
)
detachedQuadStepTrajectoryFilterBase = _detachedQuadStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 2,
    minGoodStripCharge = dict(refToPSet_ = 'SiStripClusterChargeCutLoose')
)
eras.trackingPhase1PU70.toReplaceWith(detachedQuadStepTrajectoryFilterBase,
    _detachedQuadStepTrajectoryFilterBase.clone(
        maxLostHitsFraction = 1./10.,
        constantValueForLostHitsFractionFilter = 0.501,
    )
)
detachedQuadStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('detachedQuadStepTrajectoryFilterBase'))]
)
eras.trackingPhase1PU70.toModify(detachedQuadStepTrajectoryFilter,
    filters = detachedQuadStepTrajectoryFilter.filters.value()+[cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)


import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
detachedQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'detachedQuadStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 9.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny'),
)
eras.trackingPhase1PU70.toModify(detachedQuadStepChi2Est,
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
eras.trackingPhase1PU70.toModify(detachedQuadStepTrajectoryBuilder,
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
eras.trackingPhase1PU70.toModify(detachedQuadStepTrajectoryCleanerBySharedHits,
    fractionShared = 0.095
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

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
detachedQuadStep = ClassifierMerger.clone()
detachedQuadStep.inputClassifiers=['detachedQuadStepClassifier1','detachedQuadStepClassifier2']

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
eras.trackingPhase1PU70.toReplaceWith(detachedQuadStep, RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
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

DetachedQuadStep = cms.Sequence(detachedQuadStepClusters*
                                detachedQuadStepSeedLayers*
                                detachedQuadStepSeeds*
                                detachedQuadStepTrackCandidates*
                                detachedQuadStepTracks*
                                detachedQuadStepClassifier1*detachedQuadStepClassifier2*
                                detachedQuadStep)
_DetachedQuadStep_Phase1PU70 = DetachedQuadStep.copyAndExclude([detachedQuadStepClassifier1])
_DetachedQuadStep_Phase1PU70.replace(detachedQuadStepClassifier2, detachedQuadStepSelector)
eras.trackingPhase1PU70.toReplaceWith(DetachedQuadStep, _DetachedQuadStep_Phase1PU70)
