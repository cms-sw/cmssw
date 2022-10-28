import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2RegionalShower = DQMEDAnalyzer(
    "L1TStage2RegionalShower",
    emtfSource = cms.InputTag("emtfStage2Digis"),            
#   EMTF shower unpacker ready in CMSSW_12_4_0_p2
#   Enable this module for EMTF and CSC showers
#   CSC showers are not in firmware as of 2022.04.06
#   But dummy object exists so it gives empty histograms without errors 
    cscSource = cms.InputTag("muonCSCDigis", "MuonCSCShowerDigi"), 
    monitorDir = cms.untracked.string("L1T/L1TStage2EMTF/Shower"), 
    verbose = cms.untracked.bool(False),
)

