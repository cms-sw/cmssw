import FWCore.ParameterSet.Config as cms

#
l1TrigReport = cms.EDFilter("L1TrigReport",
    L1GTReadoutRecord = cms.InputTag("gtDigis")
)


