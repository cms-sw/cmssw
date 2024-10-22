import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2Emtf = DQMEDAnalyzer(
    "L1TStage2EMTF",
    emtfSource = cms.InputTag("emtfStage2Digis"),
    monitorDir = cms.untracked.string("L1T/L1TStage2EMTF"), 
    verbose = cms.untracked.bool(False),
)

