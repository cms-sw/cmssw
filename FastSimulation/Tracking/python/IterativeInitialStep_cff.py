

import FWCore.ParameterSet.Config as cms

### ITERATIVE TRACKING: STEP 0 ###


# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeInitialSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()

iterativeInitialSeeds.outputSeedCollectionName = 'InitialPixelTriplets'
iterativeInitialSeeds.minRecHits = 3
iterativeInitialSeeds.pTMin = 0.4 # it was 0.3
iterativeInitialSeeds.maxD0 = 1.
iterativeInitialSeeds.maxZ0 = 30.
iterativeInitialSeeds.numberOfHits = 3
iterativeInitialSeeds.originRadius = 1.0 # note: standard tracking uses 0.03, but this value gives a much better agreement in rate and shape for iter0
iterativeInitialSeeds.originHalfLength = 999 # it was 15.9 
iterativeInitialSeeds.originpTMin = 0.6
iterativeInitialSeeds.zVertexConstraint = -1.0
iterativeInitialSeeds.primaryVertex = 'none'

#iterativeInitialSeeds.layerList = ['BPix1+BPix2+BPix3',
#                                   'BPix1+BPix2+FPix1_pos',
#                                   'BPix1+BPix2+FPix1_neg',
#                                   'BPix1+FPix1_pos+FPix2_pos',
#                                   'BPix1+FPix1_neg+FPix2_neg']
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
import RecoTracker.IterativeTracking.InitialStep_cff
firstStepPrimaryVerticesBeforeMixing =  RecoTracker.IterativeTracking.InitialStep_cff.firstStepPrimaryVertices.clone()

# simtrack id producer
initialStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                        trackCollection = cms.InputTag("initialStepTracks"),
                                        HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                        )

# Final selection
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepSelector,initialStep
initialStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final sequence
InitialStep = cms.Sequence(iterativeInitialSeeds
                           +initialStepTrackCandidates
                           +initialStepTracks                                    
                           +firstStepPrimaryVerticesBeforeMixing
                           +initialStepSelector
                           +initialStep
                           +initialStepSimTrackIds)




