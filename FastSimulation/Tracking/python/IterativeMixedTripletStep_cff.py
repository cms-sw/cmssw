import FWCore.ParameterSet.Config as cms

# trajectory seeds

from FastSimulation.Tracking.TrajectorySeedProducer_cfi import trajectorySeedProducer
from RecoTracker.IterativeTracking.MixedTripletStep_cff import mixedTripletStepSeedLayersA,mixedTripletStepSeedLayersB
mixedTripletStepSeeds = trajectorySeedProducer.clone(
    simTrackSelection = trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
                cms.InputTag("initialStepSimTrackIds"), 
                cms.InputTag("detachedTripletStepSimTrackIds"), 
                cms.InputTag("lowPtTripletStepSimTrackIds"), 
                cms.InputTag("pixelPairStepSimTrackIds")],
        pTMin = 0.15,
        maxD0 = 10.,
        maxZ0 = 30.
        ),
    minLayersCrossed = 3,
    originRadius = 2.0,
    originHalfLength = 10.0,
    originpTMin = 0.35,
    # combine both (A&B); Note: in FullSim, different cuts are applied for A & B seeds; 
    # in FastSim there is only one cut set, which is tuned
    # probably better to change this
    layerList = mixedTripletStepSeedLayersA.layerList+mixedTripletStepSeedLayersB.layerList
)


# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
mixedTripletStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("mixedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3)

# track producer
from RecoTracker.IterativeTracking.MixedTripletStep_cff import mixedTripletStepTracks
mixedTripletStepTracks = mixedTripletStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherThird',
    Propagator = 'PropagatorWithMaterial')

# simtrack id producer
mixedTripletStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                             trackCollection = cms.InputTag("mixedTripletStepTracks"),
                                             HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                             )


# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.IterativeTracking.MixedTripletStep_cff import mixedTripletStepSelector,mixedTripletStep

# simtrack id producer

mixedTripletStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativeMixedTripletStepTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )

# sequence
MixedTripletStep = cms.Sequence(mixedTripletStepSeeds+
                                mixedTripletStepTrackCandidates+
                                mixedTripletStepTracks+
                                mixedTripletStepSimTrackIds+
                                mixedTripletStepSelector+
                                mixedTripletStep)

