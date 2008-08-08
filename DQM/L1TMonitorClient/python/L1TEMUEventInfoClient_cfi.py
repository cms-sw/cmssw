import FWCore.ParameterSet.Config as cms

l1temuEventInfoClient = cms.EDFilter("L1TEMUEventInfoClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(1)
)


