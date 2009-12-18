import FWCore.ParameterSet.Config as cms

l1tcsctf = cms.EDFilter("L1TCSCTF",
    gmtProducer = cms.InputTag("null"), ##l1GtUnpack::DQM

    statusProducer = cms.InputTag("csctfunpacker"),
    outputFile = cms.untracked.string(''),
    lctProducer = cms.InputTag("csctfunpacker"),
    verbose = cms.untracked.bool(False),
    trackProducer = cms.InputTag("csctfunpacker"),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


