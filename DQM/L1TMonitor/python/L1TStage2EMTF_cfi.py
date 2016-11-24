import FWCore.ParameterSet.Config as cms

l1tStage2Emtf = cms.EDAnalyzer(
    "L1TStage2EMTF",
    emtfSource_daq = cms.InputTag("emtfStage2Digis"),
    emtfSource_hit = cms.InputTag("emtfStage2Digis"),
    emtfSource_track = cms.InputTag("emtfStage2Digis"),
    emtfSource_muon = cms.InputTag("emtfStage2Digis"),
    monitorDir = cms.untracked.string("L1T2016/L1TStage2EMTF"), 
    isData = cms.untracked.bool(True),
    filterBX = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
)

