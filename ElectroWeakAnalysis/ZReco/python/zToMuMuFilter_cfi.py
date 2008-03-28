import FWCore.ParameterSet.Config as cms

zToMuMuFilter = cms.EDFilter("CandCountFilter",
    src = cms.InputTag("zToMuMu"),
    minNumber = cms.uint32(1)
)


