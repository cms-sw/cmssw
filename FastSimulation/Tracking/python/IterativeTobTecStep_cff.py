import FWCore.ParameterSet.Config as cms

# trajectory seeds

from FastSimulation.Tracking.TrajectorySeedProducer_cfi import trajectorySeedProducer
from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepSeedLayersPair
tobTecStepSeeds = trajectorySeedProducer.clone( 
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
   layerList = tobTecStepSeedLayersPair.layerList
)
tobTecStepSeeds.layerList.extend(['TOB1+TOB2']) # why the extra entry?

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

# sequence
TobTecStep = cms.Sequence(tobTecStepSeeds
                          +tobTecStepTrackCandidates
                          +tobTecStepTracks
                          +tobTecStepSelector
                          +tobTecStepSimTrackIds)
