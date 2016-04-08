import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

###############################################
# Low pT and detached tracks from pixel triplets
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS

from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
_detachedTripletStepClustersBase = _trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("initialStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag(""),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)
detachedTripletStepClusters = _detachedTripletStepClustersBase.clone(
   trackClassifier                          = cms.InputTag('initialStep',"QualityMasks"),
)
eras.trackingLowPU.toReplaceWith(detachedTripletStepClusters, _detachedTripletStepClustersBase.clone(
    trajectories                             = "pixelPairStepTracks",
    oldClusterRemovalInfo                    = "pixelPairStepClusters",
    overrideTrkQuals                         = "pixelPairStepSelector:QualityMasks",
))


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
detachedTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
detachedTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('detachedTripletStepClusters')
detachedTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('detachedTripletStepClusters')

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
detachedTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
detachedTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'detachedTripletStepSeedLayers'
detachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
detachedTripletStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.3
detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.5

detachedTripletStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_detachedTripletStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
#    maxLostHitsFraction = cms.double(1./10.),
#    constantValueForLostHitsFractionFilter = cms.double(0.701),
    minimumNumberOfHits = 3,
    minPt = 0.075,
    )
detachedTripletStepTrajectoryFilterBase = _detachedTripletStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 2,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
eras.trackingLowPU.toReplaceWith(detachedTripletStepTrajectoryFilterBase, _detachedTripletStepTrajectoryFilterBase.clone(
    maxLostHitsFraction = 1./10.,
    constantValueForLostHitsFractionFilter = 0.701,
))
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
detachedTripletStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
detachedTripletStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('detachedTripletStepTrajectoryFilterBase')),
#        cms.PSet( refToPSet_ = cms.string('detachedTripletStepTrajectoryFilterShape'))
    ),
)


import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
detachedTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('detachedTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
detachedTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('detachedTripletStepTrajectoryFilter')),
    maxCand = 3,
    alwaysUseInvalidHits = True,
    estimator = cms.string('detachedTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )
eras.trackingLowPU.toModify(detachedTripletStepTrajectoryBuilder,
    maxCand = 2,
    alwaysUseInvalidHits = False,
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
detachedTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('detachedTripletStepSeeds'),
    clustersToSkip = cms.InputTag('detachedTripletStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('detachedTripletStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
detachedTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('detachedTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.13),
            allowSharedFirstHit = cms.bool(True)
            )
detachedTripletStepTrackCandidates.TrajectoryCleaner = 'detachedTripletStepTrajectoryCleanerBySharedHits'
eras.trackingLowPU.toModify(detachedTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.19)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
detachedTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('detachedTripletStep'),
    src = 'detachedTripletStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

# TRACK SELECTION AND QUALITY FLAG SETTING.


from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
detachedTripletStepClassifier1 = TrackMVAClassifierDetached.clone()
detachedTripletStepClassifier1.src = 'detachedTripletStepTracks'
detachedTripletStepClassifier1.GBRForestLabel = 'MVASelectorIter3_13TeV'
detachedTripletStepClassifier1.qualityCuts = [-0.5,0.0,0.5]
detachedTripletStepClassifier2 = TrackMVAClassifierPrompt.clone()
detachedTripletStepClassifier2.src = 'detachedTripletStepTracks'
detachedTripletStepClassifier2.GBRForestLabel = 'MVASelectorIter0_13TeV'
detachedTripletStepClassifier2.qualityCuts = [-0.2,0.0,0.4]

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
detachedTripletStep = ClassifierMerger.clone()
detachedTripletStep.inputClassifiers=['detachedTripletStepClassifier1','detachedTripletStepClassifier2']

# For LowPU
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='detachedTripletStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter3'),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedTripletStepVtxLoose',
            chi2n_par = 1.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.2, 3.0 ),
            dz_par1 = ( 1.2, 3.0 ),
            d0_par2 = ( 1.3, 3.0 ),
            dz_par2 = ( 1.3, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedTripletStepTrkLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.6, 4.0 ),
            dz_par1 = ( 1.6, 4.0 ),
            d0_par2 = ( 1.6, 4.0 ),
            dz_par2 = ( 1.6, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedTripletStepVtxTight',
            preFilterName = 'detachedTripletStepVtxLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.95, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedTripletStepTrkTight',
            preFilterName = 'detachedTripletStepTrkLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedTripletStepVtx',
            preFilterName = 'detachedTripletStepVtxTight',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.85, 3.0 ),
            dz_par1 = ( 0.8, 3.0 ),
            d0_par2 = ( 0.9, 3.0 ),
            dz_par2 = ( 0.9, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedTripletStepTrk',
            preFilterName = 'detachedTripletStepTrkTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 4,
            d0_par1 = ( 1.0, 4.0 ),
            dz_par1 = ( 1.0, 4.0 ),
            d0_par2 = ( 1.0, 4.0 ),
            dz_par2 = ( 1.0, 4.0 )
            )
        ),
    vertices = cms.InputTag("pixelVertices") #end of vpset
    ) #end of clone

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
eras.trackingLowPU.toReplaceWith(detachedTripletStep, RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('detachedTripletStepTracks'),
                                   cms.InputTag('detachedTripletStepTracks')),
    hasSelector=cms.vint32(1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("detachedTripletStepSelector","detachedTripletStepVtx"),
                                       cms.InputTag("detachedTripletStepSelector","detachedTripletStepTrk")),
    setsToMerge = cms.VPSet(cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
))

DetachedTripletStep = cms.Sequence(detachedTripletStepClusters*
                                   detachedTripletStepSeedLayers*
                                   detachedTripletStepSeeds*
                                   detachedTripletStepTrackCandidates*
                                   detachedTripletStepTracks*
                                   detachedTripletStepClassifier1*detachedTripletStepClassifier2*
                                   detachedTripletStep)
_DetachedTripletStep_LowPU = DetachedTripletStep.copyAndExclude([detachedTripletStepClassifier2])
_DetachedTripletStep_LowPU.replace(detachedTripletStepClassifier1, detachedTripletStepSelector)
eras.trackingLowPU.toReplaceWith(DetachedTripletStep, _DetachedTripletStep_LowPU)
