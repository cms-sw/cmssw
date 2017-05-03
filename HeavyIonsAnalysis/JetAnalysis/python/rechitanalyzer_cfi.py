import FWCore.ParameterSet.Config as cms

rechitanalyzer = cms.EDAnalyzer("RecHitTreeProducer",
  EBRecHitSrc = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEB"),
  EERecHitSrc = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEE"),
  BasicClusterSrc1 = cms.untracked.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
  hcalHFRecHitSrc = cms.untracked.InputTag("hfreco"),
  hcalHBHERecHitSrc = cms.untracked.InputTag("hbhereco"),
  towersSrc = cms.untracked.InputTag("towerMaker"),
  JetSrc = cms.untracked.InputTag("iterativeConePu5CaloJets"),
  useJets = cms.untracked.bool(True),
  doBasicClusters = cms.untracked.bool(False),
  doTowers = cms.untracked.bool(True),
  doEcal = cms.untracked.bool(True),
  doHcal = cms.untracked.bool(True),
  hasVtx = cms.untracked.bool(True),
  doFastJet = cms.untracked.bool(False),
  FastJetTag = cms.untracked.InputTag("kt4CaloJets"),
  doEbyEonly = cms.untracked.bool(False),
  HFtowerMin = cms.untracked.double(3),
  HFlongMin = cms.untracked.double(0.5),
  HFshortMin = cms.untracked.double(0.85),
                                doVS = cms.untracked.bool(True),
                                bkg = cms.InputTag("voronoiBackgroundCalo"),
                                etaBins = cms.int32(15),
                                fourierOrder = cms.int32(5),
                                doUEraw_ = cms.untracked.bool(False),     
                                
                                )

pfTowers = rechitanalyzer.clone(
    doEcal  = cms.untracked.bool(False),
    doHcal  = cms.untracked.bool(False),
    hasVtx  = cms.untracked.bool(False),
    doFastJet = cms.untracked.bool(False),
    towersSrc = cms.untracked.InputTag("PFTowers"),
    TowerTreePtMin = cms.untracked.double(-99),
    doVS = cms.untracked.bool(False),    
    )


rechitanalyzer.HBHETreePtMin = cms.untracked.double(15)
rechitanalyzer.HFTreePtMin = cms.untracked.double(15)
rechitanalyzer.EBTreePtMin = cms.untracked.double(15)
rechitanalyzer.EETreePtMin = cms.untracked.double(15)
rechitanalyzer.TowerTreePtMin = cms.untracked.double(-9999)
rechitanalyzer.doHF = cms.untracked.bool(True)
