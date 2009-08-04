import FWCore.ParameterSet.Config as cms

# File: caloTowers.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for caloTowers.
towerSchemeBAnalyzer = cms.EDAnalyzer(
    "CaloTowerAnalyzer",
    Debug = cms.bool(False),
    CaloTowersLabel = cms.InputTag("towerMaker"),
    FineBinning = cms.untracked.bool(False),
    FolderName = cms.untracked.string("JetMET/CaloTowers/SchemeB"),
    HLTResultsLabel = cms.InputTag("TriggerResults::HLT"),
    HLTBitLabel = cms.InputTag("HLT_MET60")
    )

towerOptAnalyzer = cms.EDAnalyzer(
    "CaloTowerAnalyzer",
    Debug = cms.bool(False),
    CaloTowersLabel = cms.InputTag("calotoweroptmaker"),
    FineBinning = cms.untracked.bool(False),
    FolderName = cms.untracked.string("JetMET/CaloTowers/Optimized"),
    HLTResultsLabel = cms.InputTag("TriggerResults::HLT"),
    HLTBitLabel = cms.InputTag("HLT_MET60")
)

