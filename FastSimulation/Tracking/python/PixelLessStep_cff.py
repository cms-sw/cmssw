import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelLessStep_cff as _standard

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
pixelLessStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(_standard.pixelLessStepClusters)

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelLessStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.pixelLessStepSeedLayers.layerList.value(),
    RegionFactoryPSet = _standard.pixelLessStepSeeds.RegionFactoryPSet,
    hitMasks = cms.InputTag("pixelLessStepMasks"),
)
pixelLessStepSeeds.seedFinderSelector.MultiHitGeneratorFactory = _standard.pixelLessStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet
pixelLessStepSeeds.seedFinderSelector.MultiHitGeneratorFactory.refitHits = False

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelLessStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("pixelLessStepSeeds"),
    MinNumberOfCrossedLayers = 6, # ?
    hitMasks = cms.InputTag("pixelLessStepMasks"),
)

# tracks
pixelLessStepTracks = _standard.pixelLessStepTracks.clone(TTRHBuilder = 'WithoutRefit')

# final selection
pixelLessStepClassifier1 = _standard.pixelLessStepClassifier1.clone()
pixelLessStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
pixelLessStepClassifier2 = _standard.pixelLessStepClassifier2.clone()
pixelLessStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"
pixelLessStep = _standard.pixelLessStep.clone()

# Final sequence 
PixelLessStep = cms.Sequence(pixelLessStepMasks
                             +pixelLessStepSeeds
                             +pixelLessStepTrackCandidates
                             +pixelLessStepTracks
                             +pixelLessStepClassifier1*pixelLessStepClassifier2
                             +pixelLessStep                             
                         )

