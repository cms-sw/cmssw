import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tcsctpg = DQMEDAnalyzer('L1TCSCTPG',
    disableROOToutput = cms.untracked.bool(True),
    csctpgSource = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


