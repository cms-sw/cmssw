import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
lowPtForwardTripletStepClusters = trackClusterRemover.clone(
    maxChi2          = 9.0,
    trajectories     = 'initialStepTracks',
    pixelClusters    = 'siPixelClusters',
    stripClusters    = 'siStripClusters',
    overrideTrkQuals = 'initialStepSelector:initialStep',
    TrackQuality     = 'highPurity',
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtForwardTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix = dict(skipClusters = cms.InputTag('lowPtForwardTripletStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('lowPtForwardTripletStepClusters')),
    layerList = ['BPix1+BPix2+FPix1_pos', 
                 'BPix1+BPix2+FPix1_neg', 
                 'BPix1+FPix1_pos+FPix2_pos', 
                 'BPix1+FPix1_neg+FPix2_neg']
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtForwardTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
	ComponentName     = 'GlobalRegionProducerFromBeamSpot',
	RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
	ptMin        = 0.2,
	originRadius = 0.03,
	nSigmaZ      = 4.0)
    ),
    OrderedHitsFactoryPSet = dict(
	SeedingLayers = 'lowPtForwardTripletStepSeedLayers'
    )
)

from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtForwardTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
lowPtForwardTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 1, ## use LostHitFraction filter instead
    minimumNumberOfHits = 3,
    minPt               = 0.1
)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
lowPtForwardTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'lowPtForwardTripletStepChi2Est',
    nSigma        = 3.0,
    MaxChi2       = 9.0
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtForwardTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter       = dict(refToPSet_ = 'lowPtForwardTripletStepTrajectoryFilter'),
    clustersToSkip         = 'lowPtForwardTripletStepClusters',
    maxCand                = 3,
    estimator              = 'lowPtForwardTripletStepChi2Est'
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtForwardTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src                       = 'lowPtForwardTripletStepSeeds',
    TrajectoryBuilderPSet     = dict(refToPSet_ = 'lowPtForwardTripletStepTrajectoryBuilder'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting          = True,
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi
lowPtForwardTripletStepTracks = RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi.TrackProducerIterativeDefault.clone(
    src           = 'lowPtForwardTripletStepTrackCandidates',
    AlgorithmName = 'lowPtTripletStep'
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
LowPtForwardTripletStepTask = cms.Task(lowPtForwardTripletStepClusters,
                                       lowPtForwardTripletStepSeedLayers,
                                       lowPtForwardTripletStepSeeds,
                                       lowPtForwardTripletStepTrackCandidates,
                                       lowPtForwardTripletStepTracks,
                                       lowPtForwardTripletStepSelector)
LowPtForwardTripletStep = cms.Sequence(LowPtForwardTripletStepTask)
