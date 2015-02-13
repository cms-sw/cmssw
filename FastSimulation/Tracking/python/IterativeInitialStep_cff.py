import FWCore.ParameterSet.Config as cms

# trajetory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeInitialSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeInitialSeeds.simTrackSelection.pTMin = 0.4
iterativeInitialSeeds.simTrackSelection.maxD0 = 1.0
iterativeInitialSeeds.simTrackSelection.maxZ0 = 999
iterativeInitialSeeds.minLayersCrossed = 3
# note: standard tracking uses for originRadius 0.03, but this value 
# gives a much better agreement in rate and shape for iter0
iterativeInitialSeeds.originpTMin = 0.6
iterativeInitialSeeds.originRadius = 1.0 
iterativeInitialSeeds.originHalfLength = 999
iterativeInitialSeeds.primaryVertex = ''

from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeInitialSeeds.layerList = PixelLayerTriplets.layerList

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
initialStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativeInitialSeeds",'InitialPixelTriplets'),
    MinNumberOfCrossedLayers = 3)

# track producer
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepTracks
initialStepTracks = initialStepTracks.clone(
    Fitter = 'KFFittingSmootherWithOutlierRejection',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial')

#vertices
from RecoTracker.IterativeTracking.InitialStep_cff import firstStepPrimaryVertices
firstStepPrimaryVertices = firstStepPrimaryVertices.clone()

# simtrack id producer
initialStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                        trackCollection = cms.InputTag("initialStepTracks"),
                                        HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                        )

# Final selection
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepSelector,initialStep

# Final sequence
InitialStep = cms.Sequence(iterativeInitialSeeds
                           +initialStepTrackCandidates
                           +initialStepTracks                                    
                           +firstStepPrimaryVertices
                           +initialStepSelector
                           +initialStep
                           +initialStepSimTrackIds)

