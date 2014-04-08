import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
lowPtForwardTripletStepClusters = cms.EDProducer("TrackClusterRemover",
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
lowPtForwardTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
lowPtForwardTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtForwardTripletStepClusters')
lowPtForwardTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtForwardTripletStepClusters')
lowPtForwardTripletStepSeedLayers.layerList = cms.vstring('BPix1+BPix2+FPix1_pos', 
                                                    'BPix1+BPix2+FPix1_neg', 
                                                    'BPix1+FPix1_pos+FPix2_pos', 
                                                    'BPix1+FPix1_neg+FPix2_neg')


# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtForwardTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.2,
    originRadius = 0.03,
    nSigmaZ = 4.0
    )
    )
    )
lowPtForwardTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtForwardTripletStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtForwardTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
lowPtForwardTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 1, ## use LostHitFraction filter instead
    minimumNumberOfHits = 3,
    minPt = 0.1
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
lowPtForwardTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('lowPtForwardTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtForwardTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('lowPtForwardTripletStepTrajectoryFilter')),
    clustersToSkip = cms.InputTag('lowPtForwardTripletStepClusters'),
    maxCand = 3,
    estimator = cms.string('lowPtForwardTripletStepChi2Est')
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtForwardTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('lowPtForwardTripletStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('lowPtForwardTripletStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtForwardTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtForwardTripletStepTrackCandidates',
    AlgorithmName = cms.string('iter1')
    )


# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtForwardTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='lowPtForwardTripletStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtForwardTripletStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtForwardTripletStepTight',
            preFilterName = 'lowPtForwardTripletStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtForwardTripletStep',
            preFilterName = 'lowPtForwardTripletStepTight',
            ),
        ) #end of vpset
    ) #end of clone

# Final sequence
LowPtForwardTripletStep = cms.Sequence(lowPtForwardTripletStepClusters*
                                       lowPtForwardTripletStepSeedLayers*
                                       lowPtForwardTripletStepSeeds*
                                       lowPtForwardTripletStepTrackCandidates*
                                       lowPtForwardTripletStepTracks*
                                       lowPtForwardTripletStepSelector)
