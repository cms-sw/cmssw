import FWCore.ParameterSet.Config as cms

rechitanalyzer = cms.EDAnalyzer("RecHitTreeProducer",
    doEbyEonly = cms.bool(False),
    doBasicClusters = cms.bool(False),
    doTowers = cms.bool(True),
                                doEcal = cms.bool(True),
                                doHBHE = cms.bool(True),
    doHF = cms.bool(True),
    doFastJet = cms.bool(False),
                                doCASTOR = cms.bool(False),
    doZDCRecHit = cms.bool(False),
    doZDCDigi = cms.bool(False),

    EBRecHitSrc = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EERecHitSrc = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    BasicClusterSrc = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
    hcalHFRecHitSrc = cms.InputTag("hfreco"),
    hcalHBHERecHitSrc = cms.InputTag("hbhereco"),
    towersSrc = cms.InputTag("towerMaker"),
    FastJetRhoTag = cms.InputTag("kt4CaloJets"),
    FastJetSigmaTag = cms.InputTag("kt4CaloJets"),
    castorRecHitSrc = cms.InputTag("castorreco"),
    zdcRecHitSrc = cms.InputTag("zdcreco"),
    zdcDigiSrc = cms.InputTag("hcalDigis"),

    nZdcTs = cms.int32(10),
    calZDCDigi = cms.bool(True),

    hasVtx = cms.bool(True),
    vtxSrc = cms.InputTag("offlinePrimaryVertices"),
    useJets = cms.bool(False),
    JetSrc = cms.InputTag("iterativeConePu5CaloJets"),
    saveBothVtx = cms.bool(False),

                                HFtowerMin = cms.untracked.double(0),
    HFlongMin = cms.untracked.double(0.5),
    HFshortMin = cms.untracked.double(0.85),     
                                HBHETreePtMin = cms.untracked.double(0),
                                HFTreePtMin = cms.untracked.double(0),
                                EBTreePtMin = cms.untracked.double(0),
                                EETreePtMin = cms.untracked.double(0),
    TowerTreePtMin = cms.untracked.double(-9999),
    )

pfTowers = rechitanalyzer.clone(
    doHF = cms.bool(False),
    hasVtx = cms.bool(False),
    towersSrc = cms.InputTag("PFTowers"),
    TowerTreePtMin = cms.untracked.double(-99),
    )

rechitanalyzerpp = rechitanalyzer.clone(
    doHF = cms.bool(False),
    vtxSrc = cms.InputTag("offlinePrimaryVertices"),
    JetSrc = cms.InputTag("ak4CaloJets"),
    )

pfTowerspp = pfTowers.clone(
    doHF = cms.bool(False),
    vtxSrc = cms.InputTag("offlinePrimaryVertices"),
    JetSrc = cms.InputTag("ak4CaloJets"),
    )
