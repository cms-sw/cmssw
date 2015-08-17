import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelLessStep_cff

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
pixelLessStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClusters)

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelLessStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    minLayersCrossed = 3,
    layerList = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeedLayers.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.RegionFactoryPSet,
    hitMasks = cms.InputTag("pixelLessStepMasks"),
)

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelLessStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("pixelLessStepSeeds"),
    MinNumberOfCrossedLayers = 6, # ?
    hitMasks = cms.InputTag("pixelLessStepMasks"),
)

# tracks
pixelLessStepTracks = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFourth',
    Propagator = 'PropagatorWithMaterial'
)
# final selection
pixelLessStepClassifier1 = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClassifier1.clone()
pixelLessStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
pixelLessStepClassifier2 = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClassifier2.clone()
pixelLessStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"
pixelLessStep = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStep.clone()

# Final sequence 
PixelLessStep = cms.Sequence(pixelLessStepMasks
                             +pixelLessStepSeeds
                             +pixelLessStepTrackCandidates
                             +pixelLessStepTracks
                             +pixelLessStepClassifier1*pixelLessStepClassifier2
                             +pixelLessStep                             
                         )

