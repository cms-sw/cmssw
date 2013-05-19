import FWCore.ParameterSet.Config as cms

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

initialStepClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution= cms.bool(True),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    doStripChargeCheck = cms.bool(True),
    stripRecHits = cms.string('siStripMatchedRecHits'),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0),
        minGoodStripCharge = cms.double(50.0)
    )
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
initialStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone(
    ComponentName = 'initialStepSeedLayers'
    )
initialStepSeedLayers.BPix.skipClusters = cms.InputTag('initialStepClusters')
initialStepSeedLayers.FPix.skipClusters = cms.InputTag('initialStepClusters')


# seeding
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
initialStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.6,
    originRadius = 0.02,
    nSigmaZ = 4.0
    )
    )
    )
initialStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'initialStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'

# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
initialStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'initialStepTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2
    )
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
initialStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('initialStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0)
)

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
initialStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'initialStepTrajectoryBuilder',
    trajectoryFilterName = 'initialStepTrajectoryFilter',
    alwaysUseInvalidHits = True,
    clustersToSkip = cms.InputTag('initialStepClusters'),
    maxCand = 6,
    estimator = cms.string('initialStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
initialStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('initialStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilder = 'initialStepTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
initialStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'initialStepTrackCandidates',
    AlgorithmName = cms.string('iter0'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
initialStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='initialStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'initialStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'initialStepTight',
            preFilterName = 'initialStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'initialStep',
            preFilterName = 'initialStepTight',
            ),
        ) #end of vpset
    ) #end of clone

# Final sequence
InitialStep = cms.Sequence(initialStepClusters*
                           initialStepSeeds*
                           initialStepTrackCandidates*
                           initialStepTracks*
                           initialStepSelector)
