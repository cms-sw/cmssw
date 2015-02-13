import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone( 
   simTrackSelection = trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("initialStepSimTrackIds"), 
            cms.InputTag("detachedTripletStepSimTrackIds"), 
            cms.InputTag("lowPtTripletStepSimTrackIds"), 
            cms.InputTag("pixelPairStepSimTrackIds"), 
            cms.InputTag("mixedTripletStepSimTrackIds"), 
            cms.InputTag("pixelLessStepSimTrackIds")],
        pTMin = 0.3,
        maxD0 = 99.,
        maxZ0 = 99.
        ),
   minLayersCrossed = 4,
   originRadius = 6.0,
   originHalfLength = 30.0,
   originpTMin = 0.6,
)
from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepSeedLayersPair as _tobTecStepSeedLayersPair
tobTecSeeds.layerList = ['TOB1+TOB2']
tobTecSeeds.layerList.extend(_tobTecStepSeedLayersPair.layerList)

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
tobTecStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("tobTecStepSeeds"),
    MinNumberOfCrossedLayers = 3)

# track producer
from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepTracks
tobTecStepTracks = tobTecStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFifth',
    Propagator = 'PropagatorWithMaterial')

# simtrack id producer
tobTecStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                       trackCollection = cms.InputTag("tobTecStepTracks"),
                                       HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                       )



# track selection
from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepSelector
tobTecStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# sequence
TobTecStep = cms.Sequence(tobTecStepSeeds
                          +tobTecStepTrackCandidates
                          +tobTecStepTracks
                          +tobTecStepSelector
                          +tobTecStepSimTrackIds)
