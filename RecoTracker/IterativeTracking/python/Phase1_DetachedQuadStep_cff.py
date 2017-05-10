import FWCore.ParameterSet.Config as cms

###############################################
# Low pT and detached tracks from pixel quadruplets
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS

from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
detachedQuadStepClusters = trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("highPtTripletStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag("highPtTripletStepClusters"),
    trackClassifier                          = cms.InputTag('highPtTripletStep',"QualityMasks"),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
detachedQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
detachedQuadStepSeedLayers.BPix.skipClusters = cms.InputTag('detachedQuadStepClusters')
detachedQuadStepSeedLayers.FPix.skipClusters = cms.InputTag('detachedQuadStepClusters')

# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
detachedQuadStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
detachedQuadStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'detachedQuadStepSeedLayers'
detachedQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
detachedQuadStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
detachedQuadStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.3
detachedQuadStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
detachedQuadStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.5
detachedQuadStepSeeds.SeedMergerPSet = cms.PSet(
    layerList = cms.PSet(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
    addRemainingTriplets = cms.bool(False),
    mergeTriplets = cms.bool(True),
    ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
)
detachedQuadStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    )
detachedQuadStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False) # FIXME: cluster check condition to be updated

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
detachedQuadStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
#    maxLostHitsFraction = cms.double(1./10.),
#    constantValueForLostHitsFractionFilter = cms.double(0.701),
    minimumNumberOfHits = 3,
    minPt = 0.075,
    maxCCCLostHits = 2,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    )
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
detachedQuadStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('detachedQuadStepTrajectoryFilterBase')),
    ),
)


import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
detachedQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('detachedQuadStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
detachedQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('detachedQuadStepTrajectoryFilter')),
    maxCand = 3,
    alwaysUseInvalidHits = True,
    estimator = cms.string('detachedQuadStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
detachedQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('detachedQuadStepSeeds'),
    clustersToSkip = cms.InputTag('detachedQuadStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('detachedQuadStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
detachedQuadStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('detachedQuadStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.13),
            allowSharedFirstHit = cms.bool(True)
            )
detachedQuadStepTrackCandidates.TrajectoryCleaner = 'detachedQuadStepTrajectoryCleanerBySharedHits'


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
detachedQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('detachedQuadStep'),
    src = 'detachedQuadStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle') # FIXME: to be updated once we get the templates to GT
    )

# TRACK SELECTION AND QUALITY FLAG SETTING.


from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
detachedQuadStepClassifier1 = TrackMVAClassifierDetached.clone()
detachedQuadStepClassifier1.src = 'detachedQuadStepTracks'
detachedQuadStepClassifier1.GBRForestLabel = 'MVASelectorIter3_13TeV'
detachedQuadStepClassifier1.qualityCuts = [-0.5,0.0,0.5]
detachedQuadStepClassifier2 = TrackMVAClassifierPrompt.clone()
detachedQuadStepClassifier2.src = 'detachedQuadStepTracks'
detachedQuadStepClassifier2.GBRForestLabel = 'MVASelectorIter0_13TeV'
detachedQuadStepClassifier2.qualityCuts = [-0.2,0.0,0.4]

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
detachedQuadStep = ClassifierMerger.clone()
detachedQuadStep.inputClassifiers=['detachedQuadStepClassifier1','detachedQuadStepClassifier2']


DetachedQuadStep = cms.Sequence(detachedQuadStepClusters*
                                detachedQuadStepSeedLayers*
                                detachedQuadStepSeeds*
                                detachedQuadStepTrackCandidates*
                                detachedQuadStepTracks*
                                detachedQuadStepClassifier1*detachedQuadStepClassifier2*
                                detachedQuadStep)
