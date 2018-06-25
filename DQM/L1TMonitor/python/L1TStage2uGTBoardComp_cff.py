import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

# Comparison of the unpacked uGT muon collections from uGT board 1 to those of boards 2 to 6.

l1tStage2uGTMuon1vsMuon2 = DQMEDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag("gtStage2Digis", "Muon"),
    muonCollection2 = cms.InputTag("gtStage2Digis", "Muon2"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard2/Muons"),
    muonCollection1Title = cms.untracked.string("Muons uGT Board 1"),
    muonCollection2Title = cms.untracked.string("Muons uGT Board 2"),
    summaryTitle = cms.untracked.string("Summary of Comparison between Muons from uGT Board 1 and Board 2"),
    verbose = cms.untracked.bool(False),
)

l1tStage2uGTMuon1vsMuon3 = l1tStage2uGTMuon1vsMuon2.clone()
l1tStage2uGTMuon1vsMuon3.muonCollection2 = cms.InputTag("gtStage2Digis", "Muon3")
l1tStage2uGTMuon1vsMuon3.muonCollection2Title = cms.untracked.string("Muons uGT Board 3")
l1tStage2uGTMuon1vsMuon3.monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard3/Muons")
l1tStage2uGTMuon1vsMuon3.summaryTitle = cms.untracked.string("Summary of Comparison between Muons from uGT Board 1 and Board 3")

l1tStage2uGTMuon1vsMuon4 = l1tStage2uGTMuon1vsMuon2.clone()
l1tStage2uGTMuon1vsMuon4.muonCollection2 = cms.InputTag("gtStage2Digis", "Muon4")
l1tStage2uGTMuon1vsMuon4.muonCollection2Title = cms.untracked.string("Muons uGT Board 4")
l1tStage2uGTMuon1vsMuon4.monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard4/Muons")
l1tStage2uGTMuon1vsMuon4.summaryTitle = cms.untracked.string("Summary of Comparison between Muons from uGT Board 1 and Board 4")

l1tStage2uGTMuon1vsMuon5 = l1tStage2uGTMuon1vsMuon2.clone()
l1tStage2uGTMuon1vsMuon5.muonCollection2 = cms.InputTag("gtStage2Digis", "Muon5")
l1tStage2uGTMuon1vsMuon5.muonCollection2Title = cms.untracked.string("Muons uGT Board 5")
l1tStage2uGTMuon1vsMuon5.monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard5/Muons")
l1tStage2uGTMuon1vsMuon5.summaryTitle = cms.untracked.string("Summary of Comparison between Muons from uGT Board 1 and Board 5")

l1tStage2uGTMuon1vsMuon6 = l1tStage2uGTMuon1vsMuon2.clone()
l1tStage2uGTMuon1vsMuon6.muonCollection2 = cms.InputTag("gtStage2Digis", "Muon6")
l1tStage2uGTMuon1vsMuon6.muonCollection2Title = cms.untracked.string("Muons uGT Board 6")
l1tStage2uGTMuon1vsMuon6.monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard6/Muons")
l1tStage2uGTMuon1vsMuon6.summaryTitle = cms.untracked.string("Summary of Comparison between Muons from uGT Board 1 and Board 6")

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
    calol2JetCollection    = cms.InputTag("gtStage2Digis", "Jet"),
    calol2EGammaCollection = cms.InputTag("gtStage2Digis", "EGamma"),
    calol2EtSumCollection  = cms.InputTag("gtStage2Digis", "EtSum"),
    calol2TauCollection    = cms.InputTag("gtStage2Digis", "Tau"),
    uGTJetCollection       = cms.InputTag("gtStage2Digis", "Jet2"),
    uGTEGammaCollection    = cms.InputTag("gtStage2Digis", "EGamma2"),
    uGTTauCollection       = cms.InputTag("gtStage2Digis", "Tau2"),
    uGTEtSumCollection     = cms.InputTag("gtStage2Digis", "EtSum2"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard2/CaloLayer2"),
)

l1tStage2uGTCalo1vsCalo3 = l1tStage2uGTCalo1vsCalo2.clone()
l1tStage2uGTCalo1vsCalo3.uGTJetCollection =    cms.InputTag("gtStage2Digis", "Jet3")
l1tStage2uGTCalo1vsCalo3.uGTEGammaCollection = cms.InputTag("gtStage2Digis", "EGamma3")
l1tStage2uGTCalo1vsCalo3.uGTTauCollection =    cms.InputTag("gtStage2Digis", "Tau3")
l1tStage2uGTCalo1vsCalo3.uGTEtSumCollection =  cms.InputTag("gtStage2Digis", "EtSum3")
l1tStage2uGTCalo1vsCalo3.monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard3/CaloLayer2")

l1tStage2uGTCalo1vsCalo4 = l1tStage2uGTCalo1vsCalo2.clone()
l1tStage2uGTCalo1vsCalo4.uGTJetCollection =    cms.InputTag("gtStage2Digis", "Jet4")
l1tStage2uGTCalo1vsCalo4.uGTEGammaCollection = cms.InputTag("gtStage2Digis", "EGamma4")
l1tStage2uGTCalo1vsCalo4.uGTTauCollection =    cms.InputTag("gtStage2Digis", "Tau4")
l1tStage2uGTCalo1vsCalo4.uGTEtSumCollection =  cms.InputTag("gtStage2Digis", "EtSum4")
l1tStage2uGTCalo1vsCalo4.monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard4/CaloLayer2")

l1tStage2uGTCalo1vsCalo5 = l1tStage2uGTCalo1vsCalo2.clone()
l1tStage2uGTCalo1vsCalo5.uGTJetCollection =    cms.InputTag("gtStage2Digis", "Jet5")
l1tStage2uGTCalo1vsCalo5.uGTEGammaCollection = cms.InputTag("gtStage2Digis", "EGamma5")
l1tStage2uGTCalo1vsCalo5.uGTTauCollection =    cms.InputTag("gtStage2Digis", "Tau5")
l1tStage2uGTCalo1vsCalo5.uGTEtSumCollection =  cms.InputTag("gtStage2Digis", "EtSum5")
l1tStage2uGTCalo1vsCalo5.monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard5/CaloLayer2")

l1tStage2uGTCalo1vsCalo6 = l1tStage2uGTCalo1vsCalo2.clone()
l1tStage2uGTCalo1vsCalo6.uGTJetCollection =    cms.InputTag("gtStage2Digis", "Jet6")
l1tStage2uGTCalo1vsCalo6.uGTEGammaCollection = cms.InputTag("gtStage2Digis", "EGamma6")
l1tStage2uGTCalo1vsCalo6.uGTTauCollection =    cms.InputTag("gtStage2Digis", "Tau6")
l1tStage2uGTCalo1vsCalo6.uGTEtSumCollection =  cms.InputTag("gtStage2Digis", "EtSum6")
l1tStage2uGTCalo1vsCalo6.monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard6/CaloLayer2")

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
