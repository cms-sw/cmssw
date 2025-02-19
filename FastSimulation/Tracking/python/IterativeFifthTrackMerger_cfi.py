import FWCore.ParameterSet.Config as cms

iterativeFifthTrackMerging = cms.EDProducer("FastTrackMerger",
                                          TrackProducers = cms.VInputTag(cms.InputTag("iterativeFifthTrackCandidatesWithPairs"),
                                                                         cms.InputTag("iterativeFifthTracksWithPairs")),
                                          RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"), #prova
                                                                                          cms.InputTag("firstfilter"),    #prova
                                                                                          cms.InputTag("secfilter"),      #prova          
                                                                                          cms.InputTag("thfilter"),      #prova          
                                                                                          cms.InputTag("foufilter")),      #prova          
                                          trackAlgo = cms.untracked.uint32(9),
                                          MinNumberOfTrajHits = cms.untracked.uint32(4),
                                          MaxLostTrajHits = cms.untracked.uint32(0)
                                          )


