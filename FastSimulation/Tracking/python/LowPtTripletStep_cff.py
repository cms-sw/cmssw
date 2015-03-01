import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.LowPtTripletStep_cff

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
lowPtTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("initialStepSimTrackIds"),
            cms.InputTag("detachedTripletStepSimTrackIds")],
        pTMin = 0.1,
        maxD0 = 5.0,
        maxZ0 = 50
    ),
    minLayersCrossed = 3,
    nSigmaZ = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.nSigmaZ,
    originpTMin = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeedLayers.layerList.value()
)

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
lowPtTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("lowPtTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3
)

# tracks
lowPtTripletStepTracks = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
)

# vertices ?

# simtrack id producer
lowPtTripletStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                             trackCollection = cms.InputTag("lowPtTripletStepTracks"),
                                             HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                         )

# final selection
lowPtTripletStepSelector = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSelector.clone()

# Final swquence 
LowPtTripletStep = cms.Sequence(lowPtTripletStepSeeds
                                +lowPtTripletStepTrackCandidates
                                +lowPtTripletStepTracks  
                                +lowPtTripletStepSelector   
                                +lowPtTripletStepSimTrackIds                            
                                )
