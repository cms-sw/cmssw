import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.InitialStep_cff

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
initialStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    minLayersCrossed = 3,
layerList = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeedLayers.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.RegionFactoryPSet,
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent")
    )

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
initialStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("initialStepSeeds"),
    MinNumberOfCrossedLayers = 3
    )

# tracks
initialStepTracks = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTracks.clone(
    Fitter = 'KFFittingSmootherWithOutlierRejection',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
    )

firstStepPrimaryVerticesBeforeMixing =  RecoTracker.IterativeTracking.InitialStep_cff.firstStepPrimaryVertices.clone()

# final selection
initialStepClassifier1 = RecoTracker.IterativeTracking.InitialStep_cff.initialStepClassifier1.clone()
initialStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
initialStepClassifier2 = RecoTracker.IterativeTracking.InitialStep_cff.initialStepClassifier2.clone()
initialStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"
initialStepClassifier3 = RecoTracker.IterativeTracking.InitialStep_cff.initialStepClassifier3.clone()
initialStepClassifier3.vertices = "firstStepPrimaryVerticesBeforeMixing"


initialStep = RecoTracker.IterativeTracking.InitialStep_cff.initialStep.clone()

# Final sequence
InitialStep = cms.Sequence(initialStepSeeds
                           +initialStepTrackCandidates
                           +initialStepTracks                                    
                           +firstStepPrimaryVerticesBeforeMixing
                           +initialStepClassifier1*initialStepClassifier2*initialStepClassifier3
                           +initialStep
                           )

