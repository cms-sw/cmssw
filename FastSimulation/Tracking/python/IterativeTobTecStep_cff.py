import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeTobTecSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeTobTecSeeds.simTrackSelection.skipSimTrackIds = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds"), 
    cms.InputTag("pixelPairStepSimTrackIds"), 
    cms.InputTag("mixedTripletStepSimTrackIds"), 
    cms.InputTag("pixelLessStepSimTrackIds")]
iterativeTobTecSeeds.simTrackSelection.pTMin = 0.3
iterativeTobTecSeeds.simTrackSelection.maxD0 = 99.
iterativeTobTecSeeds.simTrackSelection.maxZ0 = 99.
iterativeTobTecSeeds.minLayersCrossed = 4
iterativeTobTecSeeds.originRadius = 6.0
iterativeTobTecSeeds.originHalfLength = 30.0
iterativeTobTecSeeds.originpTMin = 0.6

iterativeTobTecSeeds.primaryVertex = ''

from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepSeedLayersPair
iterativeTobTecSeeds.layerList = ['TOB1+TOB2']
iterativeTobTecSeeds.layerList.extend(tobTecStepSeedLayersPair.layerList)

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
tobTecStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativeTobTecSeeds"),
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
TobTecStep = cms.Sequence(iterativeTobTecSeeds
                          +tobTecStepTrackCandidates
                          +tobTecStepTracks
                          +tobTecStepSelector
                          +tobTecStepSimTrackIds)
