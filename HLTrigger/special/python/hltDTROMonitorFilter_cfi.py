import FWCore.ParameterSet.Config as cms

dtmonitorfilter = cms.EDFilter("HLTDTROMonitorFilter",
                               inputLabel = cms.InputTag("source"))

