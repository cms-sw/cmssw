import FWCore.ParameterSet.Config as cms

# File: RecHits.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for ECAL and HCAL RecHits.
ECALAnalyzer = cms.EDAnalyzer(
    "ECALRecHitAnalyzer",
    EBRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EERecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    Debug = cms.bool(False),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("JetMET/ECALRecHits")
)
HCALAnalyzer = cms.EDAnalyzer(
    "HCALRecHitAnalyzer",
    HORecHitsLabel = cms.InputTag("horeco"),
    HBHERecHitsLabel = cms.InputTag("hbhereco"),
    Debug = cms.bool(False),
    HFRecHitsLabel = cms.InputTag("hfreco"),
    FineBinning = cms.untracked.bool(True),
    FolderName  = cms.untracked.string("JetMET/HCALRecHits")
)


