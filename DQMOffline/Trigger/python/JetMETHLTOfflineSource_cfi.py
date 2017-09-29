import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineSourceAK4 = cms.EDAnalyzer(
    "JetMETHLTOfflineSource",
    dirname = cms.untracked.string("HLT/JME/Jets/AK4"),
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
    CaloJetCollectionLabel = cms.InputTag("ak4CaloJets"),
    PFJetCollectionLabel   = cms.InputTag("ak4PFJets"),
    CaloJetCorLabel      = cms.InputTag("ak4CaloL1FastL2L3ResidualCorrector"), #dummy residual corrections now also provided for MC GTs
    PFJetCorLabel        = cms.InputTag("ak4PFL1FastL2L3ResidualCorrector"), #dummy residual corrections now also provided for MC GTs
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


jetMETHLTOfflineSourceAK8 = jetMETHLTOfflineSourceAK4.clone(
    dirname = cms.untracked.string('HLT/JME/Jets/AK8'),
    #    CaloJetCollectionLabel = cms.InputTag("ak4CaloJets"), #ak8 not available in RECO anymore, so keep ak4...
    #    PFJetCollectionLabel   = cms.InputTag("ak8PFJetsCHS"), # does not work in all matrix tests, yet
    #    PFJetCorLabel        = cms.InputTag("ak8PFCHSL1FastjetL2L3ResidualCorrector"), # does not work in all matrix tests, yet 
    PFJetCollectionLabel   = cms.InputTag("ak4PFJets"),
    PFJetCorLabel        = cms.InputTag("ak4PFL1FastL2L3ResidualCorrector"), #dummy residual corrections now also provided for MC GTs

 
    pathFilter = cms.untracked.vstring('HLT_AK8PFJet', 
    ),
    pathPairs = cms.VPSet(cms.PSet(
        denompathname = cms.string('HLT_AK8PFJet40_v'),
        pathname = cms.string('HLT_AK8PFJet60_v')
    ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJet60_v'),
            pathname = cms.string('HLT_AK8PFJet80_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJet80_v'),
            pathname = cms.string('HLT_AK8PFJet140_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJet140_v'),
            pathname = cms.string('HLT_AK8PFJet200_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJet200_v'),
            pathname = cms.string('HLT_AK8PFJet260_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJet260_v'),
            pathname = cms.string('HLT_AK8PFJet320_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJet320_v'),
            pathname = cms.string('HLT_AK8PFJet400_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJet400_v'),
            pathname = cms.string('HLT_AK8PFJet450_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJet450_v'),
            pathname = cms.string('HLT_AK8PFJet500_v')
        )),

)

jetMETHLTOfflineSourceAK8Fwd = jetMETHLTOfflineSourceAK4.clone(
    dirname = cms.untracked.string('HLT/JME/Jets/AK8Fwd'),
    PFJetCollectionLabel   = cms.InputTag("ak4PFJets"),
    PFJetCorLabel        = cms.InputTag("ak4PFL1FastL2L3ResidualCorrector"), #dummy residual corrections now also provided for MC GTs

    pathFilter = cms.untracked.vstring('HLT_AK8PFJetFwd', 
    ),
    pathPairs = cms.VPSet(cms.PSet(
        denompathname = cms.string('HLT_AK8PFJetFwd40_v'),
        pathname = cms.string('HLT_AK8PFJetFwd60_v')
    ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJetFwd60_v'),
            pathname = cms.string('HLT_AK8PFJetFwd80_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJetFwd80_v'),
            pathname = cms.string('HLT_AK8PFJetFwd140_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJetFwd140_v'),
            pathname = cms.string('HLT_AK8PFJetFwd200_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJetFwd200_v'),
            pathname = cms.string('HLT_AK8PFJetFwd260_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJetFwd260_v'),
            pathname = cms.string('HLT_AK8PFJetFwd320_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJetFwd320_v'),
            pathname = cms.string('HLT_AK8PFJetFwd400_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJetFwd400_v'),
            pathname = cms.string('HLT_AK8PFJetFwd450_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_AK8PFJetFwd450_v'),
            pathname = cms.string('HLT_AK8PFJetFwd500_v')
        )),

)

jetMETHLTOfflineSourceAK4Fwd = jetMETHLTOfflineSourceAK4.clone(
    dirname = cms.untracked.string('HLT/JME/Jets/AK4Fwd'),
    PFJetCollectionLabel   = cms.InputTag("ak4PFJets"),
    PFJetCorLabel        = cms.InputTag("ak4PFL1FastL2L3ResidualCorrector"), #dummy residual corrections now also provided for MC GTs

    pathFilter = cms.untracked.vstring('HLT_PFJetFwd', 
    ),
    pathPairs = cms.VPSet(cms.PSet(
        denompathname = cms.string('HLT_PFJetFwd40_v'),
        pathname = cms.string('HLT_PFJetFwd60_v')
    ), 
        cms.PSet(
            denompathname = cms.string('HLT_PFJetFwd60_v'),
            pathname = cms.string('HLT_PFJetFwd80_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_PFJetFwd80_v'),
            pathname = cms.string('HLT_PFJetFwd140_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_PFJetFwd140_v'),
            pathname = cms.string('HLT_PFJetFwd200_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_PFJetFwd200_v'),
            pathname = cms.string('HLT_PFJetFwd260_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_PFJetFwd260_v'),
            pathname = cms.string('HLT_PFJetFwd320_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_PFJetFwd320_v'),
            pathname = cms.string('HLT_PFJetFwd400_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_PFJetFwd400_v'),
            pathname = cms.string('HLT_PFJetFwd450_v')
        ), 
        cms.PSet(
            denompathname = cms.string('HLT_PFJetFwd450_v'),
            pathname = cms.string('HLT_PFJetFwd500_v')
        )),

)

jetMETHLTOfflineSource = cms.Sequence( jetMETHLTOfflineSourceAK4 * jetMETHLTOfflineSourceAK8 * jetMETHLTOfflineSourceAK4Fwd * jetMETHLTOfflineSourceAK8Fwd)
