import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.DetachedTripletStep_cff as _standard

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
detachedTripletStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(_standard.detachedTripletStepClusters)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
detachedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.detachedTripletStepSeedLayers.layerList.value(),
    RegionFactoryPSet = _standard.detachedTripletStepSeeds.RegionFactoryPSet,
    hitMasks = cms.InputTag("detachedTripletStepMasks")
)
detachedTripletStepSeeds.seedFinderSelector.pixelTripletGeneratorFactory = _standard.detachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
detachedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("detachedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3,
    hitMasks = cms.InputTag("detachedTripletStepMasks")
    )

# tracks 
detachedTripletStepTracks = _standard.detachedTripletStepTracks.clone(TTRHBuilder = 'WithoutRefit')

detachedTripletStepClassifier1 = _standard.detachedTripletStepClassifier1.clone()
detachedTripletStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
detachedTripletStepClassifier2 = _standard.detachedTripletStepClassifier2.clone()
detachedTripletStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"

detachedTripletStep = _standard.detachedTripletStep.clone()

# Final sequence 
DetachedTripletStep = cms.Sequence(detachedTripletStepMasks
                                   +detachedTripletStepSeeds
                                   +detachedTripletStepTrackCandidates
                                   +detachedTripletStepTracks
                                   +detachedTripletStepClassifier1*detachedTripletStepClassifier2
                                   +detachedTripletStep
                                   )
