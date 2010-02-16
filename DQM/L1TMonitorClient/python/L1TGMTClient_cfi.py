import FWCore.ParameterSet.Config as cms

l1tgmtClient = cms.EDAnalyzer("L1TGMTClient",
    input_dir = cms.untracked.string('L1T/L1TGMT'),
    prescaleLS = cms.untracked.int32(-1),
    monitorName = cms.untracked.string('L1T/L1TGMT'),
    output_dir = cms.untracked.string('L1T/L1TGMT/Client'),
    prescaleEvt = cms.untracked.int32(500)
)


