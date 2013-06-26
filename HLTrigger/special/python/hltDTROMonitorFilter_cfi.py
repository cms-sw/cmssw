import FWCore.ParameterSet.Config as cms

hltDTROMonitorFilter = cms.EDFilter("HLTDTROMonitorFilter",
  inputLabel = cms.InputTag("source")
)

