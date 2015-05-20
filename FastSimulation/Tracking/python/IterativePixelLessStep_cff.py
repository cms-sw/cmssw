import FWCore.ParameterSet.Config as cms

# step 4

# seeding
#from FastSimulation.Tracking.IterativeFourthSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelLessSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelLessSeeds.skipSimTrackIdTags = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds"), 
    cms.InputTag("pixelPairStepSimTrackIds"),  
    cms.InputTag("mixedTripletStepSimTrackIds")]
iterativePixelLessSeeds.outputSeedCollectionName = 'PixelLessPairs'
iterativePixelLessSeeds.minRecHits = 3
iterativePixelLessSeeds.pTMin = 0.3
iterativePixelLessSeeds.maxD0 = 99.
iterativePixelLessSeeds.maxZ0 = 99.
iterativePixelLessSeeds.numberOfHits = 3
iterativePixelLessSeeds.originRadius = 1.0
iterativePixelLessSeeds.originHalfLength = 12.0
iterativePixelLessSeeds.originpTMin = 0.4 # was 0.6
iterativePixelLessSeeds.zVertexConstraint = -1.0
# skip compatiblity with PV/beamspot
iterativePixelLessSeeds.skipPVCompatibility = True
iterativePixelLessSeeds.primaryVertex = 'none'

#iterativePixelLessSeeds.layerList = ['TIB1+TIB2',
#                                     'TIB1+TID1_pos','TIB1+TID1_neg',
#                                     'TID3_pos+TEC1_pos','TID3_neg+TEC1_neg',
#                                     'TID1_pos+TID2_pos','TID2_pos+TID3_pos',
#                                     'TEC1_pos+TEC2_pos','TEC2_pos+TEC3_pos','TEC3_pos+TEC4_pos','TEC3_pos+TEC5_pos','TEC4_pos+TEC5_pos',
#                                     'TID1_neg+TID2_neg','TID2_neg+TID3_neg',
#                                     'TEC1_neg+TEC2_neg','TEC2_neg+TEC3_neg','TEC3_neg+TEC4_neg','TEC3_neg+TEC5_neg','TEC4_neg+TEC5_neg']

from RecoTracker.IterativeTracking.PixelLessStep_cff import pixelLessStepSeedLayers as _pixelLessStepSeedLayers
iterativePixelLessSeeds.layerList = _pixelLessStepSeedLayers.layerList

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
pixelLessStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativePixelPairSeeds","PixelPair"),
    MinNumberOfCrossedLayers = 6 # ?
)

# track producer
from RecoTracker.IterativeTracking.PixelLessStep_cff import pixelLessStepTracks
pixelLessStepTracks = pixelLessStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFourth',
    Propagator = 'PropagatorWithMaterial'
)

# simtrack id producer
pixelLessStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                          trackCollection = cms.InputTag("pixelLessStepTracks"),
                                          HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                          )

# track selection
from RecoTracker.IterativeTracking.PixelLessStep_cff import pixelLessStepSelector,pixelLessStep
pixelLessStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# sequence
PixelLessStep = cms.Sequence(iterativePixelLessSeeds+
                             pixelLessStepTrackCandidates+
                             pixelLessStepTracks+
                             pixelLessStepSimTrackIds+
                             pixelLessStepSelector+
                             pixelLessStep)
