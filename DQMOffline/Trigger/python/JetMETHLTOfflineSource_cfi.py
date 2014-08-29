import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineSource = cms.EDAnalyzer(
    "JetMETHLTOfflineSource",
    dirname = cms.untracked.string("HLT/JetMET"),
    #
    DQMStore = cms.untracked.bool(True),
    #
    processname = cms.string("HLT"),
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    pathnameMuon = cms.untracked.vstring("HLT_IsoMu24_eta2p1_v"),
    pathnameMB = cms.untracked.vstring("HLT_Physics_v"),
    #
    verbose = cms.untracked.bool(False),
    runStandalone = cms.untracked.bool(False),
    #
    plotAll      = cms.untracked.bool(True),
    plotAllwrtMu = cms.untracked.bool(False),
    plotEff      = cms.untracked.bool(True),
    plotEffwrtMu = cms.untracked.bool(False),
    plotEffwrtMB = cms.untracked.bool(False),
    nameForEff   = cms.untracked.bool(True),
    nameForMon   = cms.untracked.bool(True),
    #
    CaloMETCollectionLabel = cms.InputTag("caloMet"),
    PFMETCollectionLabel   = cms.InputTag("pfMet"),
    #Use on-the-fly correction
    #CaloJetCollectionLabel = cms.InputTag("ak4CaloJetsL1FastL2L3"),
    #PFJetCollectionLabel   = cms.InputTag("ak4PFJetsL1FastL2L3"),
    CaloJetCollectionLabel = cms.InputTag("ak4CaloJets"),
    PFJetCollectionLabel   = cms.InputTag("ak4PFJets"),
    CaloJetCorService      = cms.string("ak4CaloL1FastL2L3"),
    PFJetCorService        = cms.string("ak4PFL1FastL2L3"),
    #
    fEMF       = cms.untracked.double(0.01),
    feta       = cms.untracked.double(2.6),
    fHPD       = cms.untracked.double(0.98),
    n90Hits    = cms.untracked.double(1),
    minNHEF    = cms.untracked.double(0.),
    maxNHEF    = cms.untracked.double(1.),
    minCHEF    = cms.untracked.double(0.),
    maxCHEF    = cms.untracked.double(1.),
    minNEMF    = cms.untracked.double(0.),
    maxNEMF    = cms.untracked.double(1.),
    minCEMF    = cms.untracked.double(0.),
    maxCEMF    = cms.untracked.double(1.),
    #
    pathFilter = cms.untracked.vstring("HLT_CaloJet",
                                       "HLT_PFJet",
                                       "HLT_PFNoPUJet",
                                       "HLT_DiPFJetAve",
                                       "HLT_PFMET",
                                       "HLT_PFchMET",                                       
                                       "HLT_MET"),
    pathRejectKeyword = cms.untracked.vstring("dEdx","NoBPTX"),
    #
    pathPairs = cms.VPSet(
        cms.PSet(
            pathname = cms.string("HLT_PFJet260_v"),
            denompathname = cms.string("HLT_PFJet40_v"),
            ),
        cms.PSet(        
            pathname = cms.string("HLT_PFNoPUJet260_v"),
            denompathname = cms.string("HLT_PFJet40_v"),
            )
        ),
    #
    JetIDParams  = cms.PSet(
        useRecHits      = cms.bool(True),
        hbheRecHitsColl = cms.InputTag("hbhereco"),
        hoRecHitsColl   = cms.InputTag("horeco"),
        hfRecHitsColl   = cms.InputTag("hfreco"),
        ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
        eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        )
)
