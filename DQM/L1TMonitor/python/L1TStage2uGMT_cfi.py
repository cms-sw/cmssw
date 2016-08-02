import FWCore.ParameterSet.Config as cms

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

l1tStage2uGMTZeroSupp = cms.EDAnalyzer(
    "L1TMP7ZeroSupp",
    fedIds = cms.vint32(1402),
    rawData = cms.InputTag("rawDataCollector"),
    # mask for inputs (pt==0 defines empty muon)
    maskCapId0 = cms.untracked.vint32(0x000001FF,
                                      0x00000000,
                                      0x000001FF,
                                      0x00000000,
                                      0x000001FF,
                                      0x00000000),
    # mask for outputs (pt==0 defines empty muon)
    maskCapId1 = cms.untracked.vint32(0x0003FC00,
                                      0x00000000,
                                      0x0003FC00,
                                      0x00000000,
                                      0x0003FC00,
                                      0x00000000),
    # no masks defined for caption IDs 2-11
    monitorDir = cms.untracked.string("L1T2016/L1TStage2uGMT/zeroSuppression"),
    verbose = cms.untracked.bool(True),
)

