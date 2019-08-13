import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2cppf = DQMEDAnalyzer('L1TStage2CPPF',
    disableROOToutput = cms.untracked.bool(True),
    rpcdigiSource = cms.InputTag("rpcCPPFRawToDigi"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


