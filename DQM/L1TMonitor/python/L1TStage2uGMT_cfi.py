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

# zero suppression check
l1tStage2uGMTZeroSupp = cms.EDAnalyzer(
    "L1TMP7ZeroSupp",
    fedIds = cms.vint32(1402),
    rawData = cms.InputTag("rawDataCollector"),
    # mask for inputs (pt==0 defines empty muon)
    maskCapId1 = cms.untracked.vint32(0x000001FF,
                                      0x00000000,
                                      0x000001FF,
                                      0x00000000,
                                      0x000001FF,
                                      0x00000000),
    # mask for outputs (pt==0 defines empty muon)
    maskCapId2 = cms.untracked.vint32(0x0003FC00,
                                      0x00000000,
                                      0x0003FC00,
                                      0x00000000,
                                      0x0003FC00,
                                      0x00000000),
    # no masks defined for caption IDs 0 and 3-11
    maxFEDReadoutSize = cms.untracked.int32(6000),
    monitorDir = cms.untracked.string("L1T2016/L1TStage2uGMT/zeroSuppression"),
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
    summaryTitle = cms.untracked.string("Summary of comparison between BMTF output muons and uGMT input muons from BMTF"),
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
    summaryTitle = cms.untracked.string("Summary of comparison between EMTF output muons and uGMT input muons from EMTF"),
    verbose = cms.untracked.bool(False),
)

