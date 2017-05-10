import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
# FIXME: detachedTriplet is dropped for time being, but may be enabled later
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
lowPtQuadStepClusters = trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
#    trajectories                             = cms.InputTag("detachedTripletStepTracks"),
    trajectories                             = cms.InputTag("detachedQuadStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
#    oldClusterRemovalInfo                    = cms.InputTag("detachedTripletStepClusters"),
#    trackClassifier                          = cms.InputTag('detachedTripletStep',"QualityMasks"),
    oldClusterRemovalInfo                    = cms.InputTag("detachedQuadStepClusters"),
    trackClassifier                          = cms.InputTag('detachedQuadStep',"QualityMasks"),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
lowPtQuadStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtQuadStepClusters')
lowPtQuadStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtQuadStepClusters')

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtQuadStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            ptMin = 0.2,
            originRadius = 0.02,
            nSigmaZ = 4.0
        )
    ),
    SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
	addRemainingTriplets = cms.bool(False),
	mergeTriplets = cms.bool(True),
	ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    )
)
lowPtQuadStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtQuadStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor
lowPtQuadStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False) # FIXME: cluster check condition to be updated


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
lowPtQuadStepStandardTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
    maxCCCLostHits = 1,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    )

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtQuadStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters   = [cms.PSet(refToPSet_ = cms.string('lowPtQuadStepStandardTrajectoryFilter')),
                 # cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))
                ]
    )

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('lowPtQuadStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('lowPtQuadStepTrajectoryFilter')),
    maxCand = 4,
    estimator = cms.string('lowPtQuadStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('lowPtQuadStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('lowPtQuadStepTrajectoryBuilder')),
    clustersToSkip = cms.InputTag('lowPtQuadStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtQuadStepTrackCandidates',
    AlgorithmName = cms.string('lowPtQuadStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle') # FIXME: to be updated once we get the templates to GT
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
lowPtQuadStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('lowPtQuadStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.16),
            allowSharedFirstHit = cms.bool(True)
            )
lowPtQuadStepTrackCandidates.TrajectoryCleaner = 'lowPtQuadStepTrajectoryCleanerBySharedHits'

# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
lowPtQuadStep =  TrackMVAClassifierPrompt.clone()
lowPtQuadStep.src = 'lowPtQuadStepTracks'
lowPtQuadStep.GBRForestLabel = 'MVASelectorIter1_13TeV'
lowPtQuadStep.qualityCuts = [-0.6,-0.3,-0.1]



# Final sequence
LowPtQuadStep = cms.Sequence(lowPtQuadStepClusters*
                             lowPtQuadStepSeedLayers*
                             lowPtQuadStepSeeds*
                             lowPtQuadStepTrackCandidates*
                             lowPtQuadStepTracks*
                             lowPtQuadStep)
