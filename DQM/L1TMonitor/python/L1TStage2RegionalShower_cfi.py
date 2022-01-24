import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2RegionalShower = DQMEDAnalyzer(
    "L1TStage2RegionalShower",
    emtfSource = cms.InputTag("emtfStage2Digis"),                   ## EMTF unpacker tag
    cscSource = cms.InputTag("muonCSCDigis", "MuonCSCShowerDigi"),  ## CSC  unpacker tag
#    emtfSource = cms.InputTag("simEmtfShowers", "EMTF"),           ## EMTF emulator tag
#    cscSource = cms.InputTag("simCscTriggerPrimitiveDigis"),       ## CSC  emulator tag
    monitorDir = cms.untracked.string("L1T/L1TStage2EMTF/Shower"), 
    verbose = cms.untracked.bool(False),
)

