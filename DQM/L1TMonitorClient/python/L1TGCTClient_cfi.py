import FWCore.ParameterSet.Config as cms

l1tGctClient = cms.EDFilter("L1TGCTClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string('L1T/L1TGCT'),
    prescaleEvt = cms.untracked.int32(1)
)


