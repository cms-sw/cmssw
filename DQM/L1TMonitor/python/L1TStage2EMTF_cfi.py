import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2Emtf = DQMEDAnalyzer(
    "L1TStage2EMTF",
    emtfSource = cms.InputTag("emtfStage2Digis"),
    monitorDir = cms.untracked.string("L1T/L1TStage2EMTF"), 
    verbose = cms.untracked.bool(False),
    isRun3 = cms.untracked.bool(False),
)

## Era: Run3_2021
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(l1tStage2Emtf, isRun3 = True)