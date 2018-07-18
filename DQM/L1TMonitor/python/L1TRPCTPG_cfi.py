import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1trpctpg = DQMEDAnalyzer('L1TRPCTPG',
    disableROOToutput = cms.untracked.bool(True),
    rpctpgSource = cms.InputTag("rpcunpacker"),
    rpctfSource = cms.InputTag("l1GtUnpack"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


