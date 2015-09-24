import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineSource = cms.EDAnalyzer(
    "JetMETHLTOfflineSource",
    dirname = cms.untracked.string("HLT/JetMET"),
    #
    processname = cms.string("HLT"),
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    pathnameMuon = cms.untracked.vstring("HLT_IsoMu24_IterTrk02_v"),
    pathnameMB = cms.untracked.vstring("HLT_Physics_v"),
    #
    verbose = cms.untracked.bool(False),
    runStandalone = cms.untracked.bool(False),
    #
    plotAll      = cms.untracked.bool(True),
    plotEff      = cms.untracked.bool(True),
    nameForEff   = cms.untracked.bool(True),
    #
    CaloMETCollectionLabel = cms.InputTag("caloMet"),
    PFMETCollectionLabel   = cms.InputTag("pfMet"),
    #Use on-the-fly correction
    #CaloJetCollectionLabel = cms.InputTag("ak4CaloJetsL1FastL2L3"),
    #PFJetCollectionLabel   = cms.InputTag("ak4PFJetsL1FastL2L3"),
    CaloJetCollectionLabel = cms.InputTag("ak4CaloJets"),
    PFJetCollectionLabel   = cms.InputTag("ak4PFJets"),
    CaloJetCorLabel      = cms.InputTag("ak4CaloL1FastL2L3Corrector"),
    PFJetCorLabel        = cms.InputTag("ak4PFL1FastL2L3Corrector"),
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
                                       "HLT_DiCaloJetAve",
                                       "HLT_PFMET",
                                       "HLT_PFchMET",
                                       "HLT_MET",
                                       "HLT_CaloMET"),
    pathRejectKeyword = cms.untracked.vstring("dEdx","NoBPTX"),
    #
    pathPairs = cms.VPSet(
        cms.PSet(
            pathname = cms.string("HLT_PFJet60_v"),
            denompathname = cms.string("HLT_PFJet40_v"),
        ),
        cms.PSet(
            pathname = cms.string("HLT_PFJet80_v"),
            denompathname = cms.string("HLT_PFJet60_v"),
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
            pathname = cms.string("HLT_PFJet450_v"),
            denompathname = cms.string("HLT_PFJet400_v"),
        ),
        cms.PSet(
            pathname = cms.string("HLT_PFJet500_v"),
            denompathname = cms.string("HLT_PFJet450_v"),
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
