import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelPairStep_cff

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
pixelPairStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepClusters)
                               
# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelPairStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    minLayersCrossed = 2,
    layerList = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet,
    hitMasks = cms.InputTag("pixelPairStepMasks"),
)
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.VertexCollection = cms.InputTag("firstStepPrimaryVerticesBeforeMixing")
# track candidate 
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelPairStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("pixelPairStepSeeds"),
    MinNumberOfCrossedLayers = 2, # ?
    hitMasks = cms.InputTag("pixelPairStepMasks"),
)

# tracks
pixelPairStepTracks = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
)
# final Selection
pixelPairStep = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStep.clone()
pixelPairStep.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final sequence 
PixelPairStep = cms.Sequence(pixelPairStepMasks
                             +pixelPairStepSeeds
                             +pixelPairStepTrackCandidates
                             +pixelPairStepTracks
                             +pixelPairStep 
                         )
