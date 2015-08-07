import FWCore.ParameterSet.Config as cms

# step 1

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelPairSeeds.skipSimTrackIdTags = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds")]
iterativePixelPairSeeds.outputSeedCollectionName = 'PixelPair'
iterativePixelPairSeeds.minRecHits =3
iterativePixelPairSeeds.pTMin = 0.3
iterativePixelPairSeeds.maxD0 = 5.
iterativePixelPairSeeds.maxZ0 = 50.
iterativePixelPairSeeds.numberOfHits = 2
iterativePixelPairSeeds.originRadius = 0.2
iterativePixelPairSeeds.originHalfLength = 17.5
iterativePixelPairSeeds.originpTMin = 0.6
iterativePixelPairSeeds.zVertexConstraint = -1.0
iterativePixelPairSeeds.primaryVertex = 'pixelVertices' # this is currently the only iteration why uses a PV instead of the BeamSpot 

#iterativePixelPairSeeds.layerList = ['BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
#                                     'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
#                                     'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 
#                                     'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
#                                     'BPix2+FPix2_pos', 'BPix2+FPix2_neg', 
#                                     'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg', 
#                                     'FPix2_pos+TEC1_pos', 'FPix2_pos+TEC2_pos', 
#                                     'FPix2_neg+TEC1_neg', 'FPix2_neg+TEC2_neg']
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSeedLayers
iterativePixelPairSeeds.layerList = pixelPairStepSeedLayers.layerList

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
pixelPairStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativePixelPairSeeds","PixelPair"),
    MinNumberOfCrossedLayers = 2 # ?
)

# track producer
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepTracks
pixelPairStepTracks = pixelPairStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherSecond',
    Propagator = 'PropagatorWithMaterial',
)

# simtrack id producer
pixelPairStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                          trackCollection = cms.InputTag("pixelPairStepTracks"),
                                          HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                          )

# Final selection
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSelector
pixelPairStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# sequence
PixelPairStep = cms.Sequence(iterativePixelPairSeeds+
                             pixelPairStepTrackCandidates+
                             pixelPairStepTracks+
                             pixelPairStepSimTrackIds+
                             pixelPairStepSelector)
