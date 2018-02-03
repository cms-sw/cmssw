import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tRct = DQMEDAnalyzer('L1TRCT',
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    HistFolder = cms.untracked.string('L1T/L1TRCT'),
    rctSource = cms.InputTag("gctDigis"),
    verbose = cms.untracked.bool(False),
    filterTriggerType = cms.int32(1),
    selectBX=cms.untracked.int32(0)
)


