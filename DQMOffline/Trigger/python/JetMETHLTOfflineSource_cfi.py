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
    plotEffwrtMC = cms.untracked.bool(True),
    nameForEff   = cms.untracked.bool(True),
    nameForMon   = cms.untracked.bool(True), 
    #
    GenCaloMETCollectionLabel = cms.InputTag("genMetCalo"),
    CaloMETCollectionLabel = cms.InputTag("met"),
    PFMETCollectionLabel   = cms.InputTag("pfMet"),
    #Use on-the-fly correction
    #CaloJetCollectionLabel = cms.InputTag("ak5CaloJetsL1FastL2L3"),
    #PFJetCollectionLabel   = cms.InputTag("ak5PFJetsL1FastL2L3"),
    CaloJetCollectionLabel = cms.InputTag("ak5CaloJets"),
    PFJetCollectionLabel   = cms.InputTag("ak5PFJets"),
    CaloJetCorService      = cms.string("ak5CaloL1FastL2L3"),
    PFJetCorService        = cms.string("ak5PFL1FastL2L3"),
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
    pathFilter = cms.untracked.vstring("HLT_Jet",
                                       "HLT_PFJet",
                                       "HLT_DiPFJetAve",
                                       "HLT_PFMET",
                                       "HLT_MET"),
    pathRejectKeyword = cms.untracked.vstring("dEdx","NoBPTX"),
    #
    pathPairs = cms.VPSet(
        cms.PSet(
         pathname = cms.string("HLT_PFJet80_v"),
         denompathname = cms.string("HLT_PFJet40_v"),
         ),
        cms.PSet(
         pathname = cms.string("HLT_PFJet140_v"),
         denompathname = cms.string("HLT_PFJet80_v"),
         ),
        cms.PSet(
         pathname = cms.string("HLT_PFJet200_v"),
         denompathname = cms.string("HLT_PFJet140_v"),
         ),
        cms.PSet(
         pathname = cms.string("HLT_PFJet260_v"),
         denompathname = cms.string("HLT_PFJet200_v"),
         ),
        cms.PSet(
         pathname = cms.string("HLT_PFJet320_v"),
         denompathname = cms.string("HLT_PFJet260_v"),
         ),
        cms.PSet(
         pathname = cms.string("HLT_PFJet400_v"),
         denompathname = cms.string("HLT_PFJet320_v"),
         ),
        cms.PSet(
         pathname = cms.string("HLT_MET120_v"),
         denompathname = cms.string("HLT_MET80_v"),
        ),
        cms.PSet(
         pathname = cms.string("HLT_MET200_v"),
         denompathname = cms.string("HLT_MET120_v"),
        ),
        cms.PSet(
         pathname = cms.string("HLT_MET300_v"),
         denompathname = cms.string("HLT_MET200_v"),  
        ),
        cms.PSet(
         pathname = cms.string("HLT_MET400_v"),
         denompathname = cms.string("HLT_MET300_v"),  
        ),
        cms.PSet(
         pathname = cms.string("HLT_PFMET180_v"),
         denompathname = cms.string("HLT_PFMET150_v"),  
        ),
        cms.PSet(
         pathname = cms.string("HLT_MET200_HBHENoiseCleaned_v"),
         denompathname = cms.string("HLT_MET120_HBHENoiseCleaned_v"),
        ),
        cms.PSet(
         pathname = cms.string("HLT_MET300_HBHENoiseCleaned_v"),
         denompathname = cms.string("HLT_MET200_HBHENoiseCleaned_v"),
        ),
        cms.PSet(
         pathname = cms.string("HLT_MET400_HBHENoiseCleaned_v"),
         denompathname = cms.string("HLT_MET300_HBHENoiseCleaned_v"),
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
