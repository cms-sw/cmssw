import FWCore.ParameterSet.Config as cms

l1tStage2EmtfEmul = cms.EDAnalyzer(
    "L1TStage2EMTF",
    emtfSource_daq = cms.InputTag("valEmtfStage2Digis", "EMTF"),
    emtfSource_hit = cms.InputTag("valEmtfStage2Digis"),
    emtfSource_track = cms.InputTag("valEmtfStage2Digis"),
    emtfSource_muon = cms.InputTag("valEmtfStage2Digis", "EMTF"),
    monitorDir = cms.untracked.string("L1T2016EMU/L1TStage2EMTFEMU"),
    isData = cms.untracked.bool(False),
    filterBX = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
)
