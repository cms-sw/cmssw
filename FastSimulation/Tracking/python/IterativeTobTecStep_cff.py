import FWCore.ParameterSet.Config as cms

# step 5

# seeding
#from FastSimulation.Tracking.IterativeFifthSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeTobTecSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeTobTecSeeds.skipSimTrackIdTags = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds"), 
    cms.InputTag("pixelPairStepSimTrackIds"), 
    cms.InputTag("mixedTripletStepSimTrackIds"), 
    cms.InputTag("pixelLessStepSimTrackIds")]
iterativeTobTecSeeds.outputSeedCollectionName = 'TobTecLayerPairs'
iterativeTobTecSeeds.minRecHits = 4
iterativeTobTecSeeds.pTMin = 0.3
iterativeTobTecSeeds.maxD0 = 99.
iterativeTobTecSeeds.maxZ0 = 99.
iterativeTobTecSeeds.numberOfHits = 2
iterativeTobTecSeeds.originRadius = 6.0 # was 5.0
iterativeTobTecSeeds.originHalfLength = 30.0 # was 10.0
iterativeTobTecSeeds.originpTMin = 0.6 # was 0.5
iterativeTobTecSeeds.zVertexConstraint = -1.0
# skip compatiblity with PV/beamspot
iterativeTobTecSeeds.skipPVCompatibility = True
iterativeTobTecSeeds.primaryVertex = 'none'

#iterativeTobTecSeeds.layerList = ['TOB1+TOB2', 
#                                  'TOB1+TEC1_pos', 'TOB1+TEC1_neg', 
#                                  'TEC1_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 
#                                  'TEC3_pos+TEC4_pos', 'TEC4_pos+TEC5_pos', 
#                                  'TEC5_pos+TEC6_pos', 'TEC6_pos+TEC7_pos', 
#                                  'TEC1_neg+TEC2_neg', 'TEC2_neg+TEC3_neg', 
#                                  'TEC3_neg+TEC4_neg', 'TEC4_neg+TEC5_neg', 
#                                  'TEC5_neg+TEC6_neg', 'TEC6_neg+TEC7_neg']
from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepSeedLayersPair as _tobTecStepSeedLayersPair
iterativeTobTecSeeds.layerList = ['TOB1+TOB2']
iterativeTobTecSeeds.layerList.extend(_tobTecStepSeedLayersPair.layerList)

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
tobTecStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativeTobTecSeeds","TobTecLayerPairs"),
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
TobTecStep = cms.Sequence(iterativeTobTecSeeds
                          +tobTecStepTrackCandidates
                          +tobTecStepTracks
                          +tobTecStepSelector
                          +tobTecStepSimTrackIds)

