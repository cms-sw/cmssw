import FWCore.ParameterSet.Config as cms

# the uGMT DQM module
l1tStage2uGMT = cms.EDAnalyzer(
    "L1TStage2uGMT",
    bmtfProducer = cms.InputTag("gmtStage2Digis", "BMTF"),
    omtfProducer = cms.InputTag("gmtStage2Digis", "OMTF"),
    emtfProducer = cms.InputTag("gmtStage2Digis", "EMTF"),
    muonProducer = cms.InputTag("gmtStage2Digis", "Muon"),
    monitorDir = cms.untracked.string("L1T2016/L1TStage2uGMT"),
    emulator = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
)

# compares the unpacked BMTF output regional muon collection with the unpacked uGMT input regional muon collection from BMTF
# only muons that do not match are filled in the histograms
l1tStage2BmtfOutVsuGMTIn = cms.EDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("bmtfDigis", "BMTF"),
    regionalMuonCollection2 = cms.InputTag("gmtStage2Digis", "BMTF"),
    monitorDir = cms.untracked.string("L1T2016/L1TStage2uGMT/BMTFoutput_vs_uGMTinput"),
    regionalMuonCollection1Title = cms.untracked.string("BMTF output data"),
    regionalMuonCollection2Title = cms.untracked.string("uGMT input data from BMTF"),
    verbose = cms.untracked.bool(False),
)

# compares the unpacked EMTF output regional muon collection with the unpacked uGMT input regional muon collection from EMTF
# only muons that do not match are filled in the histograms
l1tStage2EmtfOutVsuGMTIn = cms.EDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("emtfStage2Digis"),
    regionalMuonCollection2 = cms.InputTag("gmtStage2Digis", "EMTF"),
    monitorDir = cms.untracked.string("L1T2016/L1TStage2uGMT/EMTFoutput_vs_uGMTinput"),
    regionalMuonCollection1Title = cms.untracked.string("EMTF output data"),
    regionalMuonCollection2Title = cms.untracked.string("uGMT input data from EMTF"),
    verbose = cms.untracked.bool(False),
)

