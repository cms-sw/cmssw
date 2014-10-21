import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
highPtTripletStepClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution= cms.bool(True),
    trajectories = cms.InputTag("initialStepTracks"),
    overrideTrkQuals = cms.InputTag('initialStepSelector','initialStep'),
    TrackQuality = cms.string('highPurity'),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    )
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
highPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
highPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('highPtTripletStepClusters')
highPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('highPtTripletStepClusters')

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
highPtTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.6,
    originRadius = 0.02,
    nSigmaZ = 4.0
    )
    )
    )
highPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'highPtTripletStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
highPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
highPtTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
highPtTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('highPtTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
highPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('highPtTripletStepTrajectoryFilter')),
    maxCand = 3,
    estimator = cms.string('highPtTripletStepChi2Est')
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
highPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('highPtTripletStepSeeds'),
    clustersToSkip = cms.InputTag('highPtTripletStepClusters'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('highPtTripletStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
highPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'highPtTripletStepTrackCandidates',
    AlgorithmName = cms.string('iter1')
    )


# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
highPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='highPtTripletStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'highPtTripletStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'highPtTripletStepTight',
            preFilterName = 'highPtTripletStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'highPtTripletStep',
            preFilterName = 'highPtTripletStepTight',
            ),
        ) #end of vpset
    ) #end of clone

# Final sequence
HighPtTripletStep = cms.Sequence(highPtTripletStepClusters*
                                highPtTripletStepSeedLayers*
                                highPtTripletStepSeeds*
                                highPtTripletStepTrackCandidates*
                                highPtTripletStepTracks*
                                highPtTripletStepSelector)
