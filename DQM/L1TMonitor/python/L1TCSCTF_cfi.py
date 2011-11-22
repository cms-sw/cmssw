import FWCore.ParameterSet.Config as cms

l1tcsctf = cms.EDAnalyzer("L1TCSCTF",
    gmtProducer = cms.InputTag("l1GtUnpack::DQM"),

    statusProducer = cms.InputTag("csctfunpacker"),
    outputFile = cms.untracked.string(''),
    lctProducer = cms.InputTag("csctfunpacker"),
    verbose = cms.untracked.bool(False),
    trackProducer = cms.InputTag("csctfunpacker"),
    mbProducer = cms.InputTag("csctfunpacker:DT"),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


