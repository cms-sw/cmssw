import FWCore.ParameterSet.Config as cms

l1tdttpg = cms.EDAnalyzer("L1TDTTPG",
    disableROOToutput = cms.untracked.bool(True),
    dttpgSource = cms.InputTag("l1tdttpgunpack","","DQM"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


