import FWCore.ParameterSet.Config as cms

jetMETAnalyzer = cms.EDAnalyzer("JetMETAnalyzer",

    #
    # Output files
    #
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('jetMETMonitoring.root'),

    #
    #
    #
    TriggerResultsLabel        = cms.InputTag("TriggerResults::HLT"),
    processname                = cms.string("HLT"),
    JetLo                      = cms.string("HLT_Jet30"),
    JetHi                      = cms.string("HLT_Jet110"),

    #
    # Jet-related
    #                                                                   
    DoPFJetAnalysis            = cms.untracked.bool(True),
    DoJPTJetAnalysis           = cms.untracked.bool(True),
    DoJetAnalysis              = cms.untracked.bool(True),
    DoJetCleaning              = cms.untracked.bool(True),
    DoJetPtAnalysis            = cms.untracked.bool(False),                           
    PFJetsCollectionLabel      = cms.InputTag("iterativeCone5PFJets"),
    JPTJetsCollectionLabel     = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    SCJetsCollectionLabel      = cms.InputTag("sisCone5CaloJets"),
    ICJetsCollectionLabel      = cms.InputTag("iterativeCone5CaloJets"),

    #
    # MET-related
    #                                                                   
    DoCaloMETAnalysis            = cms.untracked.bool(True),
    DoTcMETAnalysis              = cms.untracked.bool(True),
    DoPfMETAnalysis              = cms.untracked.bool(True),
    DoHTMHTAnalysis              = cms.untracked.bool(True),

    #
    # For caloMETAnalysis "met"
    #
    caloMETAnalysis = cms.PSet(
    CaloMETCollectionLabel = cms.InputTag("met"),
    Source = cms.string("CaloMET"),
    HLTPathsJetMB = cms.vstring(),
#   When it is empty, it accepts all the triggers
#   HLTPathsJetMB = cms.vstring(
#                 "HLT_L1Jet15","HLT_Jet30","HLT_Jet50","HLT_Jet80","HLT_Jet110","HLT_Jet180",
#                 "HLT_DiJetAve15","HLT_DiJetAve30","HLT_DiJetAve50","HLT_DiJetAve70",
#                 "HLT_DiJetAve130","HLT_DiJetAve220",
#                 "HLT_MinBias"),
    HLT_HighPtJet = cms.string("HLT_Jet50U"),
    HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
    HLT_HighMET   = cms.string("HLT_MET100"),
    HLT_LowMET    = cms.string("HLT_L1MET20"),
    HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
    HLT_Muon      = cms.string("HLT_Mu9"),
    CaloTowersLabel    = cms.InputTag("towerMaker"),         
    JetCollectionLabel = cms.InputTag("sisCone5CaloJets"),   # jets used for event cleanup
    JetIDParams = cms.PSet(
    useRecHits = cms.bool(True),
    hbheRecHitsColl = cms.InputTag("hbhereco"),
    hoRecHitsColl   = cms.InputTag("horeco"),
    hfRecHitsColl   = cms.InputTag("hfreco"),
    ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
    ),
    HcalNoiseRBXCollection  = cms.InputTag("hcalnoise"),
    HcalNoiseSummary        = cms.InputTag("hcalnoise"),
    HighPtJetThreshold = cms.double(60.),
    LowPtJetThreshold  = cms.double(15.),
    HighMETThreshold = cms.double(110.),
    LowMETThreshold  = cms.double(30.),
    verbose     = cms.int32(0),
    etThreshold = cms.double(1.),                            # default MET threshold
    allHist     = cms.bool(True),
    allSelection= cms.bool(False)
    ),

    #
    # For caloMETAnalysis "metNoHF"
    #
    caloMETNoHFAnalysis = cms.PSet(
    CaloMETCollectionLabel   = cms.InputTag("metNoHF"),
    Source = cms.string("CaloMETNoHF"),
    HLTPathsJetMB = cms.vstring(),
    HLT_HighPtJet = cms.string("HLT_Jet50U"),
    HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
    HLT_HighMET   = cms.string("HLT_MET100"),
    HLT_LowMET    = cms.string("HLT_L1MET20"),
    HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
    HLT_Muon      = cms.string("HLT_Mu9"),
    CaloTowersLabel    = cms.InputTag("towerMaker"),
    JetCollectionLabel = cms.InputTag("sisCone5CaloJets"),   # jets used for event cleanup
    JetIDParams = cms.PSet(
    useRecHits = cms.bool(True),
    hbheRecHitsColl = cms.InputTag("hbhereco"),
    hoRecHitsColl   = cms.InputTag("horeco"),
    hfRecHitsColl   = cms.InputTag("hfreco"),
    ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
    ),
    HcalNoiseRBXCollection  = cms.InputTag("hcalnoise"),
    HcalNoiseSummary        = cms.InputTag("hcalnoise"),
    HighPtJetThreshold = cms.double(60.),
    LowPtJetThreshold  = cms.double(15.),
    HighMETThreshold = cms.double(110.),
    LowMETThreshold  = cms.double(30.),
    verbose     = cms.int32(0),
    etThreshold = cms.double(1.),
    allHist     = cms.bool(False),
    allSelection= cms.bool(False)
    ),

    #
    # For caloMETAnalysis "metHO"
    #
    caloMETHOAnalysis = cms.PSet(
    CaloMETCollectionLabel   = cms.InputTag("metHO"),
    Source = cms.string("CaloMETHO"),
    HLTPathsJetMB = cms.vstring(),
    HLT_HighPtJet = cms.string("HLT_Jet50U"),
    HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
    HLT_HighMET   = cms.string("HLT_MET100"),
    HLT_LowMET    = cms.string("HLT_L1MET20"),
    HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
    HLT_Muon      = cms.string("HLT_Mu9"),
    CaloTowersLabel    = cms.InputTag("towerMaker"),
    JetCollectionLabel = cms.InputTag("sisCone5CaloJets"),   # jets used for event cleanup
    JetIDParams = cms.PSet(
    useRecHits = cms.bool(True),
    hbheRecHitsColl = cms.InputTag("hbhereco"),
    hoRecHitsColl   = cms.InputTag("horeco"),
    hfRecHitsColl   = cms.InputTag("hfreco"),
    ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
    ),
    HcalNoiseRBXCollection  = cms.InputTag("hcalnoise"),
    HcalNoiseSummary        = cms.InputTag("hcalnoise"),
    HighPtJetThreshold = cms.double(60.),
    LowPtJetThreshold  = cms.double(15.),
    HighMETThreshold = cms.double(110.),
    LowMETThreshold  = cms.double(30.),
    verbose     = cms.int32(0),
    etThreshold = cms.double(1.),
    allHist     = cms.bool(False),
    allSelection= cms.bool(False)
    ),

    #
    # For caloMETAnalysis
    #
    caloMETNoHFHOAnalysis = cms.PSet(
    CaloMETCollectionLabel   = cms.InputTag("metNoHFHO"),
    Source = cms.string("CaloMETNoHFHO"),
    HLTPathsJetMB = cms.vstring(),
    HLT_HighPtJet = cms.string("HLT_Jet50U"),
    HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
    HLT_HighMET   = cms.string("HLT_MET100"),
    HLT_LowMET    = cms.string("HLT_L1MET20"),
    HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
    HLT_Muon      = cms.string("HLT_Mu9"),
    CaloTowersLabel    = cms.InputTag("towerMaker"),
    JetCollectionLabel = cms.InputTag("sisCone5CaloJets"),   # jets used for event cleanup
    JetIDParams = cms.PSet(
    useRecHits = cms.bool(True),
    hbheRecHitsColl = cms.InputTag("hbhereco"),
    hoRecHitsColl   = cms.InputTag("horeco"),
    hfRecHitsColl   = cms.InputTag("hfreco"),
    ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
    ),
    HcalNoiseRBXCollection  = cms.InputTag("hcalnoise"),
    HcalNoiseSummary        = cms.InputTag("hcalnoise"),
    HighPtJetThreshold = cms.double(60.),
    LowPtJetThreshold  = cms.double(15.),
    HighMETThreshold = cms.double(110.),
    LowMETThreshold  = cms.double(30.),
    verbose     = cms.int32(0),
    etThreshold = cms.double(1.),
    allHist     = cms.bool(False),
    allSelection= cms.bool(False)
    ),

    #
    # For tcMETAnalysis
    #
    tcMETAnalysis = cms.PSet(
    verbose     = cms.int32(0),
    TcMETCollectionLabel         = cms.InputTag("tcMet"),
    Source = cms.string("TcMET"),
    HLTPathsJetMB = cms.vstring(),
    etThreshold = cms.double(1.)
    ),

    #
    # For pfMETAnalysis
    #
    pfMETAnalysis = cms.PSet(
    verbose     = cms.int32(0),
    PfMETCollectionLabel         = cms.InputTag("pfMet"),
    Source = cms.string("PfMET"),
    HLTPathsJetMB = cms.vstring(),
    etThreshold = cms.double(1.)
    ),

    #
    # For HTMHTAnalysis
    #
    HTMHTAnalysis = cms.PSet(
    verbose     = cms.int32(0),
    JetCollectionForHTMHTLabel   = cms.InputTag("sisCone5CaloJets"),
    Source = cms.string("HTMHT"),
    HLTPathsJetMB = cms.vstring(),
    ptThreshold = cms.double(20.)
    ),

    #
    # For jetAnalysis
    #
    jetAnalysis = cms.PSet(
        ptThreshold = cms.double(3.),
        fEM = cms.double(-1),
        N90Cells = cms.double(0),
        eBin    = cms.int32(100),
        phiMin  = cms.double(-3.2),
        ptBin   = cms.int32(100),
        eMin    = cms.double(0.0),
        eMax    = cms.double(500.0),
        pMin    = cms.double(0.0),
        etaBin  = cms.int32(100),
        etaMin  = cms.double(-5.0),
        ptMin   = cms.double(0.0),
        phiBin  = cms.int32(70),
        pBin    = cms.int32(100),
        ptMax   = cms.double(50.0),
        etaMax  = cms.double(5.0),
        pMax    = cms.double(500.0),
        phiMax  = cms.double(3.2)
    ),

    #
    # For jetcleaning Analysis
    #
    CleanedjetAnalysis = cms.PSet(
        ptThreshold = cms.double(10.),
        fEM = cms.double(0.01),
        N90Cells = cms.double(2),
        eBin    = cms.int32(100),
        phiMin  = cms.double(-3.2),
        ptBin   = cms.int32(100),
        eMin    = cms.double(0.0),
        eMax    = cms.double(500.0),
        pMin    = cms.double(0.0),
        etaBin  = cms.int32(100),
        etaMin  = cms.double(-5.0),
        ptMin   = cms.double(0.0),
        phiBin  = cms.int32(70),
        pBin    = cms.int32(100),
        ptMax   = cms.double(50.0),
        etaMax  = cms.double(5.0),
        pMax    = cms.double(500.0),
        phiMax  = cms.double(3.2)
    ),

    #
    # For Pt jet Analysis
    #
    PtAnalysis = cms.PSet(    
        phiMin  = cms.double(-3.2),
        phiMax  = cms.double(3.2),
        phiBin  = cms.int32(70),
        ptMin   = cms.double(0.0),
        ptMax   = cms.double(200.0),
        ptBin   = cms.int32(200),
        etaBin  = cms.int32(100),
        etaMin  = cms.double(-5.0),             
        etaMax  = cms.double(5.0)
    ),

    #
    # For PF jetAnalysis
    #
    pfJetAnalysis = cms.PSet(
        ptThreshold = cms.double(3.),
        fEM = cms.double(-1),
        N90Cells = cms.double(0),
        eBin    = cms.int32(100),
        phiMin  = cms.double(-3.2),
        ptBin   = cms.int32(100),
        eMin    = cms.double(0.0),
        eMax    = cms.double(500.0),
        pMin    = cms.double(0.0),
        etaBin  = cms.int32(100),
        etaMin  = cms.double(-5.0),
        ptMin   = cms.double(0.0),
        phiBin  = cms.int32(70),
        pBin    = cms.int32(100),
        ptMax   = cms.double(50.0),
        etaMax  = cms.double(5.0),
        pMax    = cms.double(500.0),
        phiMax  = cms.double(3.2)
    ),

    #
    # For JPT jetAnalysis
    #
    JPTJetAnalysis = cms.PSet(
        ptThreshold = cms.double(3.),
        fEM = cms.double(-1),
        N90Cells = cms.double(0),
        eBin    = cms.int32(100),
        phiMin  = cms.double(-3.2),
        ptBin   = cms.int32(100),
        eMin    = cms.double(0.0),
        eMax    = cms.double(500.0),
        pMin    = cms.double(0.0),
        etaBin  = cms.int32(100),
        etaMin  = cms.double(-5.0),
        ptMin   = cms.double(0.0),
        phiBin  = cms.int32(70),
        pBin    = cms.int32(100),
        ptMax   = cms.double(50.0),
        etaMax  = cms.double(5.0),
        pMax    = cms.double(500.0),
        phiMax  = cms.double(3.2)
    )


)
