import FWCore.ParameterSet.Config as cms

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
muonSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
muonSeeds.selectMuons = True
muonSeeds.seedingAlgo = ['MuonSeeded']
muonSeeds.minRecHits = [3] # placeholder; how much should it be?
muonSeeds.pTMin = [2.0]
muonSeeds.maxD0 = [999.]
muonSeeds.maxZ0 = [999.]
muonSeeds.numberOfHits = [3] # placeholder; how much should it be?
muonSeeds.originRadius = [999.]
muonSeeds.originHalfLength = [999.] 
muonSeeds.originpTMin = [2.0] 
muonSeeds.zVertexConstraint = [-1.0]
muonSeeds.primaryVertices = ['none']

# candidate producer
import FastSimulation.Tracking.TrackCandidateProducer_cfi
muonSeededCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
muonSeededCandidates.SeedProducer = cms.InputTag("muonSeeds","MuonSeeded")
muonSeededCandidates.TrackProducers = ['initialStep','lowPtTripletStep','pixelPairStep','detachedTripletStepTracks','mixedTripletStepTracks','pixelLessStepTracks','tobTecStep']
muonSeededCandidates.MinNumberOfCrossedLayers = 3 # placeholder; how much should it be?

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
muonSeededTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
muonSeededTracks.src = 'muonSeededCandidates'
muonSeededTracks.TTRHBuilder = 'WithoutRefit'
muonSeededTracks.Fitter = 'KFFittingSmootherFifth'
muonSeededTracks.Propagator = 'PropagatorWithMaterial'


# track merger
#from FastSimulation.Tracking.IterativeFifthTrackMerger_cfi import *
muonSeededTracksOutIn = cms.EDProducer("FastTrackMerger",
                                  TrackProducers = cms.VInputTag(cms.InputTag("muonSeededCandidates"),
                                                                 cms.InputTag("muonSeededTracks")),
                                  RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("initialStepTracks"),
                                                                                  cms.InputTag("lowPtTripletStepTracks"),  
                                                                                  cms.InputTag("pixelPairStepTracks"),  
                                                                                  cms.InputTag("detachedTripletStepTracks"),    
                                                                                  cms.InputTag("mixedTripletStepTracks"),     
                                                                                  cms.InputTag("pixelLessStepTracks"),   
                                                                                  cms.InputTag("tobTecStepTracks")),   
                                  trackAlgo = cms.untracked.uint32(13), # iter9 
                                  MinNumberOfTrajHits = cms.untracked.uint32(5), # placeholder; how much should it be?
                                  MaxLostTrajHits = cms.untracked.uint32(8) # placeholder; how much should it be?
                                  )
muonSeededTracksInOut = cms.EDProducer("FastTrackMerger", # notice that this is exactly the same as above, apart from iteration number; the two will differ only at the final selection stage
                                  TrackProducers = cms.VInputTag(cms.InputTag("muonSeededCandidates"),
                                                                 cms.InputTag("muonSeededTracks")),
                                  RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("initialStepTracks"),
                                                                                  cms.InputTag("lowPtTripletStepTracks"),  
                                                                                  cms.InputTag("pixelPairStepTracks"),  
                                                                                  cms.InputTag("detachedTripletStepTracks"),    
                                                                                  cms.InputTag("mixedTripletStepTracks"),     
                                                                                  cms.InputTag("pixelLessStepTracks"),   
                                                                                  cms.InputTag("tobTecStepTracks")),   
                                  trackAlgo = cms.untracked.uint32(14), # iter10 
                                  MinNumberOfTrajHits = cms.untracked.uint32(5), # placeholder; how much should it be?
                                  MaxLostTrajHits = cms.untracked.uint32(8) # placeholder; how much should it be?
                                  )

# track selection
from RecoTracker.IterativeTracking.MuonSeededStep_cff import muonSeededTracksInOutSelector 
from RecoTracker.IterativeTracking.MuonSeededStep_cff import muonSeededTracksOutInSelector 


muonSeededStep = cms.Sequence(
    muonSeeds +
    muonSeededCandidates +
    muonSeededTracks+
    muonSeededTracksOutIn+
    muonSeededTracksInOut+
    muonSeededTracksInOutSelector+
    muonSeededTracksOutInSelector
    )
