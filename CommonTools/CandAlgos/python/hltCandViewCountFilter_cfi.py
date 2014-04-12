import FWCore.ParameterSet.Config as cms

hltCandViewCountFilter = cms.EDFilter("CandViewCountFilter",
   src       = cms.InputTag("hltCollection"),
   minNumber = cms.uint32(0)
)
