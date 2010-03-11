import FWCore.ParameterSet.Config as cms

dimuonsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("userDataDimuons"),
    minNumber = cms.uint32(1)
)


