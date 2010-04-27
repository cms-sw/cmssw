import FWCore.ParameterSet.Config as cms

iterativeThirdTrackMerging = cms.EDProducer("FastTrackMerger",
                                          TrackProducers = cms.VInputTag(cms.InputTag("iterativeThirdTrackCandidatesWithPairs"),
                                                                         cms.InputTag("iterativeThirdTracksWithPairs")),
                                          RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"), #prova
                                                                                          cms.InputTag("firstfilter"),    #prova
                                                                                          cms.InputTag("secfilter")),     #prova
                                          trackAlgo = cms.untracked.uint32(7),
                                          MinNumberOfTrajHits = cms.untracked.uint32(4),
                                          MaxLostTrajHits = cms.untracked.uint32(0)                                          
                                          )


