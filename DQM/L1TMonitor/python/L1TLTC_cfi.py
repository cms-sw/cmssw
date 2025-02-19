import FWCore.ParameterSet.Config as cms

l1tLtc = cms.EDAnalyzer("L1TLTC",
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


