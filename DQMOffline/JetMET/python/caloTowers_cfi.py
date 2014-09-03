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
    AllHist = cms.untracked.bool(True),
    FolderName = cms.untracked.string("JetMET/CaloTowers/SchemeB"),
    HBHENoiseFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHENoiseFilterResult"),
    HLTSelection = cms.untracked.bool(False),
    HLTResultsLabel = cms.InputTag("TriggerResults::HLT"),
    HLTBitLabels = cms.VInputTag(
    cms.InputTag("HLT_MET100"),
    cms.InputTag("HLT_HT100U")
    )
    )

