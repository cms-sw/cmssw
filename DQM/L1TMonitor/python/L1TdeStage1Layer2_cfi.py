import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1TdeStage1Layer2 = DQMEDAnalyzer('L1TdeGCT',
    DataEmulCompareSource = cms.InputTag("l1compareforstage1"),
    HistFolder = cms.untracked.string('L1TEMU/Stage1Layer2expert'),
    HistFile = cms.untracked.string(''),
    disableROOToutput = cms.untracked.bool(True),
    VerboseFlag = cms.untracked.int32(0),
    stage1_layer2_ = cms.bool(True)
)


