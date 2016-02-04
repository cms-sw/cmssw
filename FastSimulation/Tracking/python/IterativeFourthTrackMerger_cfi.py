import FWCore.ParameterSet.Config as cms

iterativeFourthTrackMerging = cms.EDProducer("FastTrackMerger",
TrackProducers = cms.VInputTag(cms.InputTag("iterativeFourthTrackCandidatesWithPairs"),
                               cms.InputTag("iterativeFourthTracksWithPairs")),
RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"), #prova
                                                cms.InputTag("firstfilter"),    #prova
                                                cms.InputTag("secfilter"),      #prova          
                                                cms.InputTag("thfilter")),      #prova          
trackAlgo = cms.untracked.uint32(8),
MinNumberOfTrajHits = cms.untracked.uint32(5),
MaxLostTrajHits = cms.untracked.uint32(0)
)


