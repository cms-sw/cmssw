import FWCore.ParameterSet.Config as cms

l1tdttpg = cms.EDAnalyzer("L1TDTTPG",
    disableROOToutput = cms.untracked.bool(True),
    dttpgSource = cms.InputTag("muonDTDigis"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


