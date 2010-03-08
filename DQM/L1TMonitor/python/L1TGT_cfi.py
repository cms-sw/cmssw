import FWCore.ParameterSet.Config as cms

l1tgt = cms.EDAnalyzer("L1TGT",
    DQMStore = cms.untracked.bool(True),
    gtEvmSource = cms.InputTag("l1GtEvmUnpack","","DQM"),
    gtSource = cms.InputTag("l1GtUnpack","","DQM"),
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)


