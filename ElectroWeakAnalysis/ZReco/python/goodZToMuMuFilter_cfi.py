import FWCore.ParameterSet.Config as cms

goodZToMuMuFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodZToMuMu"),
    minNumber = cms.uint32(1)
)


