import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.InitialStep_cff

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
initialStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.4,
        maxD0 = 1.0,
        maxZ0 = -1,
        ),
    minLayersCrossed = 3,
    nSigmaZ = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.RegionFactoryPSet.RegionPSet.nSigmaZ,
    ptMin = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeedLayers.layerList.value()
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
initialStepSelector = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSelector.clone()
initialStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"
initialStep = RecoTracker.IterativeTracking.InitialStep_cff.initialStep.clone()

# Final sequence
InitialStep = cms.Sequence(initialStepSeeds
                           +initialStepTrackCandidates
                           +initialStepTracks                                    
                           +firstStepPrimaryVerticesBeforeMixing
                           +initialStepSelector
                           +initialStep
                           )

