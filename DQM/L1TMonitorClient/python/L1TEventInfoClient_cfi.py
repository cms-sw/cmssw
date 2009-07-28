import FWCore.ParameterSet.Config as cms

l1tEventInfoClient = cms.EDFilter("L1TEventInfoClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(1),
    dataMaskedSystems = cms.untracked.vstring("empty"),
    emulMaskedSystems = cms.untracked.vstring("empty")
)


