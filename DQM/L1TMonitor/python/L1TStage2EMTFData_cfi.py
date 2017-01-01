import FWCore.ParameterSet.Config as cms

l1tStage2EmtfData = cms.EDAnalyzer(
    "L1TStage2EMTF",
    emtfSource = cms.InputTag("emtfStage2Digis"),
    monitorDir = cms.untracked.string("L1T2016EMU/L1TStage2EMTFData"), 
    isData = cms.untracked.bool(True),
    filterBX = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
)

