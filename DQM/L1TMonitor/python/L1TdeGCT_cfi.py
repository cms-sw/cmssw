import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1TdeGCT = DQMEDAnalyzer('L1TdeGCT',
    DataEmulCompareSource = cms.InputTag("l1compare"),
    HistFolder = cms.untracked.string('L1TEMU/GCTexpert'),
    HistFile = cms.untracked.string(''),
    disableROOToutput = cms.untracked.bool(True),
    VerboseFlag = cms.untracked.int32(0),
    stage1_layer2_ = cms.bool(False)
)


