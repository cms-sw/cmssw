import FWCore.ParameterSet.Config as cms

l1tcsctpg = cms.EDFilter("L1TCSCTPG",
    disableROOToutput = cms.untracked.bool(True),
    csctpgSource = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi","DQM"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


