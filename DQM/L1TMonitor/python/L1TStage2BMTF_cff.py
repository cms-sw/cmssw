import FWCore.ParameterSet.Config as cms

# the BMTF DQM module
from DQM.L1TMonitor.L1TStage2BMTF_cfi import *

# zero suppression DQM
l1tStage2BmtfZeroSupp = cms.EDAnalyzer(
    "L1TMP7ZeroSupp",
    fedIds = cms.vint32(1376, 1377),
    rawData = cms.InputTag("rawDataCollector"),
    # mask for inputs (pt==0 defines empty muon)
    maskCapId1 = cms.untracked.vint32(0x01c00000,
                                      0x01c00000,
                                      0x01c00000,
                                      0x01c00000,
                                      0x00000000,
                                      0x00000000),
    # mask for outputs (pt==0 defines empty muon)
    maskCapId2 = cms.untracked.vint32(0x000001FF,
                                      0x00000000,
                                      0x000001FF,
                                      0x00000000,
                                      0x000001FF,
                                      0x00000000),
    # no masks defined for caption IDs 0 and 3-11
    maxFEDReadoutSize = cms.untracked.int32(7000),
    monitorDir = cms.untracked.string("L1T/L1TStage2BMTF/zeroSuppression/AllEvts"),
    verbose = cms.untracked.bool(False),
)

# ZS of validation events (to be used after fat event filter)
l1tStage2BmtfZeroSuppFatEvts = l1tStage2BmtfZeroSupp.clone()
l1tStage2BmtfZeroSuppFatEvts.monitorDir = cms.untracked.string("L1T/L1TStage2BMTF/zeroSuppression/FatEvts")
l1tStage2BmtfZeroSuppFatEvts.maxFEDReadoutSize = cms.untracked.int32(25000)

# sequences
l1tStage2BmtfOnlineDQMSeq = cms.Sequence(
    l1tStage2Bmtf +
    l1tStage2BmtfZeroSupp
)

l1tStage2BmtfValidationEventOnlineDQMSeq = cms.Sequence(
    l1tStage2BmtfZeroSuppFatEvts
)
