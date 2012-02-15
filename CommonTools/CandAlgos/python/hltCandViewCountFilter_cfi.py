import FWCore.ParameterSet.Config as cms

hltCandViewCountFilter = cms.EDFilter("CandViewCountFilter",
   src    = cms.InputTag("hltCollection"),
   minNumber = cms.unit32(0)
)
