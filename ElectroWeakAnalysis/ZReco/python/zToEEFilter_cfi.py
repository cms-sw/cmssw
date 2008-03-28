import FWCore.ParameterSet.Config as cms

zToEEFilter = cms.EDFilter("CandCountFilter",
    src = cms.InputTag("zToEE"),
    minNumber = cms.uint32(1)
)


