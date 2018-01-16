import FWCore.ParameterSet.Config as cms

l1tcsctpg = DQMStep1Module('L1TCSCTPG',
    disableROOToutput = cms.untracked.bool(True),
    csctpgSource = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


