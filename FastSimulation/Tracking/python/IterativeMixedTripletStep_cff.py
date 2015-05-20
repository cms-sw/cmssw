import FWCore.ParameterSet.Config as cms

# step 3

# seeding
#from FastSimulation.Tracking.IterativeMixedTripletStepSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeMixedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeMixedTripletStepSeeds.skipSimTrackIdTags = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds"), 
    cms.InputTag("pixelPairStepSimTrackIds")]
iterativeMixedTripletStepSeeds.outputSeedCollectionName = 'MixedTriplets'
iterativeMixedTripletStepSeeds.minRecHits = 3
iterativeMixedTripletStepSeeds.pTMin = 0.15
iterativeMixedTripletStepSeeds.maxD0 = 10.
iterativeMixedTripletStepSeeds.maxZ0 = 30.
iterativeMixedTripletStepSeeds.numberOfHits = 3
iterativeMixedTripletStepSeeds.originRadius = 2.0 # was 1.2
iterativeMixedTripletStepSeeds.originHalfLength = 10.0 # was 7.0
iterativeMixedTripletStepSeeds.originpTMin = 0.35 # we need to add another seed for endcaps only, with 0.5
iterativeMixedTripletStepSeeds.zVertexConstraint = -1.0
iterativeMixedTripletStepSeeds.primaryVertex = 'none'

#iterativeMixedTripletStepSeeds.layerList = ['BPix1+BPix2+BPix3',
#                                            'BPix1+BPix2+FPix1_pos',
#                                            'BPix1+BPix2+FPix1_neg',
#                                            'BPix1+FPix1_pos+FPix2_pos',
#                                            'BPix1+FPix1_neg+FPix2_neg']
from RecoTracker.IterativeTracking.MixedTripletStep_cff import mixedTripletStepSeedLayersA as _mixedTripletStepSeedLayersA ,mixedTripletStepSeedLayersB as _mixedTripletStepSeedLayersB
# combine both (A&B); Note: in FullSim, different cuts are applied for A & B seeds; in FastSim cuts are tuned (no need to corresponded to FullSim values)
iterativeMixedTripletStepSeeds.layerList = _mixedTripletStepSeedLayersA.layerList+_mixedTripletStepSeedLayersB.layerList

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
mixedTripletStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# sequence
MixedTripletStep = cms.Sequence(iterativeMixedTripletStepSeeds+
                                mixedTripletStepTrackCandidates+
                                mixedTripletStepTracks+
                                mixedTripletStepSimTrackIds+
                                mixedTripletStepSelector+
                                mixedTripletStep)

