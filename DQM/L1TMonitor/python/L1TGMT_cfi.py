import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tGmt = DQMEDAnalyzer('L1TGMT',
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    gmtSource = cms.InputTag("l1GtUnpack"),
    DQMStore = cms.untracked.bool(True)
)


