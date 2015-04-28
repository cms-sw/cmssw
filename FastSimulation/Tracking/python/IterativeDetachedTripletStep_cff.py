import FWCore.ParameterSet.Config as cms

### ITERATIVE TRACKING: STEP 3 ###

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeDetachedTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeDetachedTripletSeeds.skipSimTrackIdTags = [cms.InputTag("initialStepSimTrackIds")]
iterativeDetachedTripletSeeds.outputSeedCollectionName = 'DetachedPixelTriplets'
iterativeDetachedTripletSeeds.minRecHits = 3
iterativeDetachedTripletSeeds.pTMin = 0.3
iterativeDetachedTripletSeeds.maxD0 = 30. # it was 5.
iterativeDetachedTripletSeeds.maxZ0 = 50.
iterativeDetachedTripletSeeds.numberOfHits = 3
iterativeDetachedTripletSeeds.originRadius = 1.5
iterativeDetachedTripletSeeds.originHalfLength = 15.
iterativeDetachedTripletSeeds.originpTMin = 0.075
iterativeDetachedTripletSeeds.zVertexConstraint = -1.0
iterativeDetachedTripletSeeds.primaryVertex = 'none'

#iterativeDetachedTripletSeeds.layerList = ['BPix1+BPix2+BPix3',
#                                   'BPix1+BPix2+FPix1_pos',
#                                   'BPix1+BPix2+FPix1_neg',
#                                   'BPix1+FPix1_pos+FPix2_pos',
#                                   'BPix1+FPix1_neg+FPix2_neg']
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeDetachedTripletSeeds.layerList = PixelLayerTriplets.layerList

# candidate producer
#from FastSimulation.Tracking.IterativeSecondCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
detachedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativeDetachedTripletSeeds",'DetachedPixelTriplets'),
    MinNumberOfCrossedLayers = 3)

# track producer
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepTracks
detachedTripletStepTracks = detachedTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    Propagator = 'PropagatorWithMaterial',
    TTRHBuilder = 'WithoutRefit'
)

# simtrack id producer
detachedTripletStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("detachedTripletStepTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )


# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepSelector,detachedTripletStep
detachedTripletStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# sequence
DetachedTripletStep = cms.Sequence(iterativeDetachedTripletSeeds+
                                            detachedTripletStepTrackCandidates+
                                            detachedTripletStepTracks+
                                            detachedTripletStepSimTrackIds+
                                            detachedTripletStepSelector+
                                            detachedTripletStep)
