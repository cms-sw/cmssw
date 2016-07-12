import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.MixedTripletStep_cff as _standard

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
mixedTripletStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(_standard.mixedTripletStepClusters)
mixedTripletStepMasks.oldHitRemovalInfo = cms.InputTag("pixelPairStepMasks")

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
mixedTripletStepSeedsA = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.mixedTripletStepSeedLayersA.layerList.value(),
    RegionFactoryPSet = _standard.mixedTripletStepSeedsA.RegionFactoryPSet,
    hitMasks = cms.InputTag("mixedTripletStepMasks")
)
mixedTripletStepSeedsA.seedFinderSelector.pixelTripletGeneratorFactory = _standard.mixedTripletStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet


###
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
mixedTripletStepSeedsB = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.mixedTripletStepSeedLayersB.layerList.value(),
    RegionFactoryPSet = _standard.mixedTripletStepSeedsB.RegionFactoryPSet,
    hitMasks = cms.InputTag("mixedTripletStepMasks")
)
mixedTripletStepSeedsB.seedFinderSelector.pixelTripletGeneratorFactory = _standard.mixedTripletStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet

mixedTripletStepSeeds = _standard.mixedTripletStepSeeds.clone()

#track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
mixedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("mixedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3,
    hitMasks = cms.InputTag("mixedTripletStepMasks"),
)

# tracks
mixedTripletStepTracks = _standard.mixedTripletStepTracks.clone(TTRHBuilder = 'WithoutRefit')

# final selection
mixedTripletStepClassifier1 = _standard.mixedTripletStepClassifier1.clone()
mixedTripletStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
mixedTripletStepClassifier2 = _standard.mixedTripletStepClassifier2.clone()
mixedTripletStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"

mixedTripletStep = _standard.mixedTripletStep.clone()

# Final sequence 
MixedTripletStep =  cms.Sequence(mixedTripletStepMasks
                                 +mixedTripletStepSeedsA
                                 +mixedTripletStepSeedsB
                                 +mixedTripletStepSeeds
                                 +mixedTripletStepTrackCandidates
                                 +mixedTripletStepTracks
                                 +mixedTripletStepClassifier1*mixedTripletStepClassifier2
                                 +mixedTripletStep                                 
                             )
