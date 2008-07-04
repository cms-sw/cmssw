import FWCore.ParameterSet.Config as cms

l1tdeEcalClient = cms.EDFilter("L1TdeECALClient",
    prescaleLS = cms.untracked.int32(-1),
    verbose = cms.untracked.bool(False),
    prescaleEvt = cms.untracked.int32(500),
    monitorDir = cms.untracked.string('L1TEMU/xpert/Ecal/')
)


