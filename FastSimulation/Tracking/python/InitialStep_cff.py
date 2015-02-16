import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.InitialStep_cff

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
initialStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.4,
        maxD0 = 1.0,
        maxZ0 = 999,
        ),
    minLayersCrossed = 3,
    nSigmaZ = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.nSigmaZ,
    originpTMin = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.ptMin,
    originRadius = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.originRadius,
    layerList = RecoTracker.IterativeTracking.InitialStep_cff.initialstepseedlayers.layerList.clone()
    )

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
initialStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("initialStepSeeds"),
    MinNumberOfCrossedLayers = 3
    )

# tracks
initialStepTracks = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTracks.clone(
    Fitter = 'KFFittingSmootherWithOutlierRejection',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
    )

# vertices
firstStepPrimaryVertices = RecoTracker.IterativeTracking.InitialStep_cff.firstStepPrimaryVertices.clone()

# simtrack id producer
initialStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                        trackCollection = cms.InputTag("initialStepTracks"),
                                        HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                        )

# final selection
initialStepSelector = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSelector.clone()
initialStep = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSelector.initialStep.clone()

# Final sequence
InitialStep = cms.Sequence(initialStepSeeds
                           +initialStepTrackCandidates
                           +initialStepTracks                                    
                           +firstStepPrimaryVertices
                           +initialStepSelector
                           +initialStep
                           +initialStepSimTrackIds
                           )

