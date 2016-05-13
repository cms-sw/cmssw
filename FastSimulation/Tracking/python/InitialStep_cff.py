import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.InitialStep_cff as _standard

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
initialStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.initialStepSeedLayers.layerList.value(),
    RegionFactoryPSet = _standard.initialStepSeeds.RegionFactoryPSet
)
initialStepSeeds.seedFinderSelector.pixelTripletGeneratorFactory = _standard.initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet
initialStepSeeds.seedFinderSelector.pixelTripletGeneratorFactory.SeedComparitorPSet.ComponentName = "none"

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
initialStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("initialStepSeeds"),
    MinNumberOfCrossedLayers = 3
    )

# tracks
initialStepTracks = _standard.initialStepTracks.clone(TTRHBuilder = 'WithoutRefit')

firstStepPrimaryVerticesBeforeMixing =  _standard.firstStepPrimaryVertices.clone()

# final selection
initialStepClassifier1 = _standard.initialStepClassifier1.clone()
initialStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
initialStepClassifier2 = _standard.initialStepClassifier2.clone()
initialStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"
initialStepClassifier3 = _standard.initialStepClassifier3.clone()
initialStepClassifier3.vertices = "firstStepPrimaryVerticesBeforeMixing"


initialStep = _standard.initialStep.clone()

# Final sequence
InitialStep = cms.Sequence(initialStepSeeds
                           +initialStepTrackCandidates
                           +initialStepTracks                                    
                           +firstStepPrimaryVerticesBeforeMixing
                           +initialStepClassifier1*initialStepClassifier2*initialStepClassifier3
                           +initialStep
                           )

