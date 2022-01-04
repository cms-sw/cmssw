import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

# Comparison of the unpacked uGT muon collections from uGT board 1 to those of boards 2 to 6.

l1tStage2uGTMuon1vsMuon2 = DQMEDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag("gtStage2Digis", "Muon"),
    muonCollection2 = cms.InputTag("gtStage2Digis", "Muon2"),
    muonCollection1Title = cms.untracked.string("Muons uGT Board 1"),
    muonCollection2Title = cms.untracked.string("Muons uGT Board 2"),
    summaryTitle = cms.untracked.string("Summary of Comparison between Muons from uGT Board 1 and uGT Board 2"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard2/Muons"),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGTMuon1vsMuon3 = l1tStage2uGTMuon1vsMuon2.clone(
    muonCollection2 = "gtStage2Digis:Muon3",
    muonCollection2Title = "Muons uGT Board 3",
    summaryTitle = "Summary of Comparison between Muons from uGT Board 1 and uGT Board 3",
    monitorDir = "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard3/Muons"
)
l1tStage2uGTMuon1vsMuon4 = l1tStage2uGTMuon1vsMuon2.clone(
    muonCollection2 = "gtStage2Digis:Muon4",
    muonCollection2Title = "Muons uGT Board 4",
    summaryTitle = "Summary of Comparison between Muons from uGT Board 1 and uGT Board 4",
    monitorDir = "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard4/Muons"
)
l1tStage2uGTMuon1vsMuon5 = l1tStage2uGTMuon1vsMuon2.clone(
    muonCollection2 = "gtStage2Digis:Muon5",
    muonCollection2Title = "Muons uGT Board 5",
    summaryTitle = "Summary of Comparison between Muons from uGT Board 1 and uGT Board 5",
    monitorDir = "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard5/Muons"
)
l1tStage2uGTMuon1vsMuon6 = l1tStage2uGTMuon1vsMuon2.clone(
    muonCollection2 = "gtStage2Digis:Muon6",
    muonCollection2Title = "Muons uGT Board 6",
    summaryTitle = "Summary of Comparison between Muons from uGT Board 1 and uGT Board 6",
    monitorDir = "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard6/Muons"
)
l1tStage2uGTBoardCompMuonsSeq = cms.Sequence(
    l1tStage2uGTMuon1vsMuon2 +
    l1tStage2uGTMuon1vsMuon3 +
    l1tStage2uGTMuon1vsMuon4 +
    l1tStage2uGTMuon1vsMuon5 +
    l1tStage2uGTMuon1vsMuon6
)

# Comparison of the unpacked uGT CaloLayer2 collections from uGT board 1 to those of boards 2 to 6.

l1tStage2uGTCalo1vsCalo2 = DQMEDAnalyzer(
    "L1TStage2uGTCaloLayer2Comp",
    collection1Title  = cms.untracked.string("uGT Board 1"),
    collection2Title  = cms.untracked.string("uGT Board 2"),
    JetCollection1    = cms.InputTag("gtStage2Digis", "Jet"),
    JetCollection2    = cms.InputTag("gtStage2Digis", "Jet2"),
    EGammaCollection1 = cms.InputTag("gtStage2Digis", "EGamma"),
    EGammaCollection2 = cms.InputTag("gtStage2Digis", "EGamma2"),
    TauCollection1    = cms.InputTag("gtStage2Digis", "Tau"),
    TauCollection2    = cms.InputTag("gtStage2Digis", "Tau2"),
    EtSumCollection1  = cms.InputTag("gtStage2Digis", "EtSum"),
    EtSumCollection2  = cms.InputTag("gtStage2Digis", "EtSum2"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard2/CaloLayer2"),
)

l1tStage2uGTCalo1vsCalo3 = l1tStage2uGTCalo1vsCalo2.clone(
    collection2Title  = "uGT Board 3", 
    JetCollection2    = "gtStage2Digis:Jet3",
    EGammaCollection2 = "gtStage2Digis:EGamma3",
    TauCollection2    = "gtStage2Digis:Tau3",
    EtSumCollection2  = "gtStage2Digis:EtSum3",
    monitorDir = "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard3/CaloLayer2"
)
l1tStage2uGTCalo1vsCalo4 = l1tStage2uGTCalo1vsCalo2.clone(
    collection2Title  = "uGT Board 4", 
    JetCollection2    = "gtStage2Digis:Jet4",
    EGammaCollection2 = "gtStage2Digis:EGamma4",
    TauCollection2    = "gtStage2Digis:Tau4",
    EtSumCollection2  = "gtStage2Digis:EtSum4",
    monitorDir = "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard4/CaloLayer2"
)
l1tStage2uGTCalo1vsCalo5 = l1tStage2uGTCalo1vsCalo2.clone(
    collection2Title  = "uGT Board 5", 
    JetCollection2    = "gtStage2Digis:Jet5",
    EGammaCollection2 = "gtStage2Digis:EGamma5",
    TauCollection2    = "gtStage2Digis:Tau5",
    EtSumCollection2  = "gtStage2Digis:EtSum5",
    monitorDir = "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard5/CaloLayer2"
)
l1tStage2uGTCalo1vsCalo6 = l1tStage2uGTCalo1vsCalo2.clone(
    collection2Title  = "uGT Board 6", 
    JetCollection2    = "gtStage2Digis:Jet6",
    EGammaCollection2 = "gtStage2Digis:EGamma6",
    TauCollection2    = "gtStage2Digis:Tau6",
    EtSumCollection2  = "gtStage2Digis:EtSum6",
    monitorDir = "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard6/CaloLayer2"
)
l1tStage2uGTBoardCompCaloLayer2Seq = cms.Sequence(
    l1tStage2uGTCalo1vsCalo2 +
    l1tStage2uGTCalo1vsCalo3 +
    l1tStage2uGTCalo1vsCalo4 +
    l1tStage2uGTCalo1vsCalo5 +
    l1tStage2uGTCalo1vsCalo6
)

l1tStage2uGTBoardCompSeq = cms.Sequence(
    l1tStage2uGTBoardCompMuonsSeq +
    l1tStage2uGTBoardCompCaloLayer2Seq
)
