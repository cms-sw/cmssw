import FWCore.ParameterSet.Config as cms

l1tGmt = cms.EDAnalyzer("L1TGMT",
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    gmtSource = cms.InputTag("gtDigis"),
    DQMStore = cms.untracked.bool(True)
)


