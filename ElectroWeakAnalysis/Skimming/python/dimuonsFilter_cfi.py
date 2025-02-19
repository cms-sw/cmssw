import FWCore.ParameterSet.Config as cms

dimuonsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuons"),
    minNumber = cms.uint32(1)
)


