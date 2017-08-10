import FWCore.ParameterSet.Config as cms

# the uGMT DQM module
from DQM.L1TMonitor.L1TStage2uGMT_cfi import *

# the uGMT intermediate muon DQM modules
l1tStage2uGMTIntermediateBMTF = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("gmtStage2Digis", "imdMuonsBMTF"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/intermediate_muons/BMTF"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from BMTF "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateOMTFNeg = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("gmtStage2Digis", "imdMuonsOMTFNeg"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/intermediate_muons/OMTF_neg"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from OMTF neg. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateOMTFPos = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("gmtStage2Digis", "imdMuonsOMTFPos"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/intermediate_muons/OMTF_pos"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from OMTF pos. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateEMTFNeg = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("gmtStage2Digis", "imdMuonsEMTFNeg"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/intermediate_muons/EMTF_neg"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from EMTF neg. "),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTIntermediateEMTFPos = cms.EDAnalyzer(
    "L1TStage2uGMTMuon",
    muonProducer = cms.InputTag("gmtStage2Digis", "imdMuonsEMTFPos"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/intermediate_muons/EMTF_pos"),
    titlePrefix = cms.untracked.string("uGMT intermediate muon from EMTF pos. "),
    verbose = cms.untracked.bool(False),
)

# zero suppression DQM
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
    maskCapId2 = cms.untracked.vint32(0x0007FC00,
                                      0x00000000,
                                      0x0007FC00,
                                      0x00000000,
                                      0x0007FC00,
                                      0x00000000),
    # mask for validation event outputs (pt==0 defines empty muon)
    maskCapId3 = cms.untracked.vint32(0x0007FC00,
                                      0x00000000,
                                      0x0007FC00,
                                      0x00000000,
                                      0x0007FC00,
                                      0x00000000),
    # no masks defined for caption IDs 0 and 4-11
    maxFEDReadoutSize = cms.untracked.int32(9000),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/zeroSuppression/AllEvts"),
    verbose = cms.untracked.bool(False),
)

# ZS of validation events (to be used after fat event filter)
l1tStage2uGMTZeroSuppFatEvts = l1tStage2uGMTZeroSupp.clone()
l1tStage2uGMTZeroSuppFatEvts.monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/zeroSuppression/FatEvts")

# List of bins to ignore
ignoreBins = [1]

# compares the unpacked BMTF output regional muon collection with the unpacked uGMT input regional muon collection from BMTF
# only muons that do not match are filled in the histograms
l1tStage2BmtfOutVsuGMTIn = cms.EDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("bmtfDigis", "BMTF"),
    regionalMuonCollection2 = cms.InputTag("gmtStage2Digis", "BMTF"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/BMTFoutput_vs_uGMTinput"),
    regionalMuonCollection1Title = cms.untracked.string("BMTF output data"),
    regionalMuonCollection2Title = cms.untracked.string("uGMT input data from BMTF"),
    summaryTitle = cms.untracked.string("Summary of comparison between BMTF output muons and uGMT input muons from BMTF"),
    ignoreBin = cms.untracked.vint32(ignoreBins),
    verbose = cms.untracked.bool(False),
)

# compares the unpacked EMTF output regional muon collection with the unpacked uGMT input regional muon collection from EMTF
# only muons that do not match are filled in the histograms
l1tStage2EmtfOutVsuGMTIn = cms.EDAnalyzer(
    "L1TStage2RegionalMuonCandComp",
    regionalMuonCollection1 = cms.InputTag("emtfStage2Digis"),
    regionalMuonCollection2 = cms.InputTag("gmtStage2Digis", "EMTF"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/EMTFoutput_vs_uGMTinput"),
    regionalMuonCollection1Title = cms.untracked.string("EMTF output data"),
    regionalMuonCollection2Title = cms.untracked.string("uGMT input data from EMTF"),
    summaryTitle = cms.untracked.string("Summary of comparison between EMTF output muons and uGMT input muons from EMTF"),
    verbose = cms.untracked.bool(False),
)

# compares the unpacked uGMT muon collection to the unpacked uGT muon collection
# only muons that do not match are filled in the histograms
l1tStage2uGMTOutVsuGTIn = cms.EDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag("gmtStage2Digis", "Muon"),
    muonCollection2 = cms.InputTag("gtStage2Digis", "Muon"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/uGMToutput_vs_uGTinput"),
    muonCollection1Title = cms.untracked.string("uGMT output muons"),
    muonCollection2Title = cms.untracked.string("uGT input muons"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT output muons and uGT input muons"),
    verbose = cms.untracked.bool(False),
)

# The five modules below compare the primary unpacked uGMT muon collection to goes to uGT board 0
# to the unpacked uGMT muon collections that are sent to uGT boards 1 to 5.
# Only muons that do not match are filled in the histograms
l1tStage2uGMTMuonVsuGMTMuonCopy1 = cms.EDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag("gmtStage2Digis", "Muon"),
    muonCollection2 = cms.InputTag("gmtStage2Digis", "MuonCopy1"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1"),
    muonCollection1Title = cms.untracked.string("uGMT muons"),
    muonCollection2Title = cms.untracked.string("uGMT muons copy 1"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT muon copy 1"),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGMTMuonVsuGMTMuonCopy2 = l1tStage2uGMTMuonVsuGMTMuonCopy1.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy2.muonCollection2 = cms.InputTag("gmtStage2Digis", "MuonCopy2")
l1tStage2uGMTMuonVsuGMTMuonCopy2.monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2")
l1tStage2uGMTMuonVsuGMTMuonCopy2.muonCollection2Title = cms.untracked.string("uGMT muons copy 2")
l1tStage2uGMTMuonVsuGMTMuonCopy2.summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT muon copy 2")

l1tStage2uGMTMuonVsuGMTMuonCopy3 = l1tStage2uGMTMuonVsuGMTMuonCopy1.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy3.muonCollection2 = cms.InputTag("gmtStage2Digis", "MuonCopy3")
l1tStage2uGMTMuonVsuGMTMuonCopy3.monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3")
l1tStage2uGMTMuonVsuGMTMuonCopy3.muonCollection2Title = cms.untracked.string("uGMT muons copy 3")
l1tStage2uGMTMuonVsuGMTMuonCopy3.summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT muon copy 3")

l1tStage2uGMTMuonVsuGMTMuonCopy4 = l1tStage2uGMTMuonVsuGMTMuonCopy1.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy4.muonCollection2 = cms.InputTag("gmtStage2Digis", "MuonCopy4")
l1tStage2uGMTMuonVsuGMTMuonCopy4.monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4")
l1tStage2uGMTMuonVsuGMTMuonCopy4.muonCollection2Title = cms.untracked.string("uGMT muons copy 4")
l1tStage2uGMTMuonVsuGMTMuonCopy4.summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT muon copy 4")

l1tStage2uGMTMuonVsuGMTMuonCopy5 = l1tStage2uGMTMuonVsuGMTMuonCopy1.clone()
l1tStage2uGMTMuonVsuGMTMuonCopy5.muonCollection2 = cms.InputTag("gmtStage2Digis", "MuonCopy5")
l1tStage2uGMTMuonVsuGMTMuonCopy5.monitorDir = cms.untracked.string("L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5")
l1tStage2uGMTMuonVsuGMTMuonCopy5.muonCollection2Title = cms.untracked.string("uGMT muons copy 5")
l1tStage2uGMTMuonVsuGMTMuonCopy5.summaryTitle = cms.untracked.string("Summary of comparison between uGMT muons and uGMT muon copy 5")

# sequences
l1tStage2uGMTOnlineDQMSeq = cms.Sequence(
    l1tStage2uGMT +
    l1tStage2uGMTIntermediateBMTF +
    l1tStage2uGMTIntermediateOMTFNeg +
    l1tStage2uGMTIntermediateOMTFPos +
    l1tStage2uGMTIntermediateEMTFNeg +
    l1tStage2uGMTIntermediateEMTFPos +
    l1tStage2uGMTZeroSupp +
    l1tStage2BmtfOutVsuGMTIn +
    l1tStage2EmtfOutVsuGMTIn +
    l1tStage2uGMTOutVsuGTIn
)

l1tStage2uGMTValidationEventOnlineDQMSeq = cms.Sequence(
    l1tStage2uGMTZeroSuppFatEvts +
    l1tStage2uGMTMuonVsuGMTMuonCopy1 +
    l1tStage2uGMTMuonVsuGMTMuonCopy2 +
    l1tStage2uGMTMuonVsuGMTMuonCopy3 +
    l1tStage2uGMTMuonVsuGMTMuonCopy4 +
    l1tStage2uGMTMuonVsuGMTMuonCopy5
)
