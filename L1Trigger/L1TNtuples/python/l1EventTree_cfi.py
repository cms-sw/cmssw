import FWCore.ParameterSet.Config as cms

l1EventTree = cms.EDAnalyzer("L1EventTreeProducer",
                             hltSource            = cms.InputTag("TriggerResults::HLT"),
                             puMCFile             = cms.untracked.string(""),
                             puDataFile           = cms.untracked.string(""),
                             puMCHist             = cms.untracked.string(""),
                             puDataHist           = cms.untracked.string(""),                            
                             useAvgVtx            = cms.untracked.bool(True),
                             maxAllowedWeight     = cms.untracked.double(-1)
)
