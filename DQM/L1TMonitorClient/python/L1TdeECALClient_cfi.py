import FWCore.ParameterSet.Config as cms

l1tdeEcalClient = cms.EDFilter("L1TdeECALClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string('L1T/L1DEMonEcal'),
    prescaleEvt = cms.untracked.int32(500)
)


