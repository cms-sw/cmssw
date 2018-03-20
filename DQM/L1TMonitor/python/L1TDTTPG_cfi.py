import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdttpg = DQMEDAnalyzer('L1TDTTPG',
    disableROOToutput = cms.untracked.bool(True),
    dttpgSource = cms.InputTag("muonDTDigis"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


