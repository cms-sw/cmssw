import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeMixedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeMixedTripletStepSeeds.simTrackSelection.skipSimTrackIds = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds"), 
    cms.InputTag("pixelPairStepSimTrackIds")]
iterativeMixedTripletStepSeeds.simTrackSelection.pTMin = 0.15
iterativeMixedTripletStepSeeds.simTrackSelection.maxD0 = 10.
iterativeMixedTripletStepSeeds.simTrackSelection.maxZ0 = 30.
iterativeMixedTripletStepSeeds.minLayersCrossed = 3
iterativeMixedTripletStepSeeds.originRadius = 2.0
iterativeMixedTripletStepSeeds.originHalfLength = 10.0
iterativeMixedTripletStepSeeds.originpTMin = 0.35
iterativeMixedTripletStepSeeds.primaryVertex = ''
# combine both (A&B); Note: in FullSim, different cuts are applied for A & B seeds; 
# in FastSim there is only one cut set, which is tuned
# probably better to change this
from RecoTracker.IterativeTracking.MixedTripletStep_cff import mixedTripletStepSeedLayersA,mixedTripletStepSeedLayersB
iterativeMixedTripletStepSeeds.layerList = mixedTripletStepSeedLayersA.layerList+mixedTripletStepSeedLayersB.layerList

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
mixedTripletStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativeMixedTripletStepSeeds","MixedTriplets"),
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
MixedTripletStep = cms.Sequence(iterativeMixedTripletStepSeeds+
                                mixedTripletStepTrackCandidates+
                                mixedTripletStepTracks+
                                mixedTripletStepSimTrackIds+
                                mixedTripletStepSelector+
                                mixedTripletStep)

