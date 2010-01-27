import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jptDQMConfig_cff import *

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
    DoPFJetCleaning            = cms.untracked.bool(True),

    DoJPTJetAnalysis           = cms.untracked.bool(True),
    DoJPTJetCleaning           = cms.untracked.bool(True),

    DoJetAnalysis              = cms.untracked.bool(True),
    DoJetCleaning              = cms.untracked.bool(True),
    DoIterativeCone            = cms.untracked.bool(False),

    DoJetPtAnalysis            = cms.untracked.bool(False),                           
    DoJetPtCleaning            = cms.untracked.bool(False),                           

    PFJetsCollectionLabel      = cms.InputTag("iterativeCone5PFJets"),
    JPTJetsCollectionLabel     = cms.InputTag("ak5CaloJets"),
    #JPTJetsCollectionLabel     = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    SCJetsCollectionLabel      = cms.InputTag("sisCone5CaloJets"),
    AKJetsCollectionLabel      = cms.InputTag("ak5CaloJets"),
    ICJetsCollectionLabel      = cms.InputTag("iterativeCone5CaloJets"),

    #
    # MET-related
    #                                                                   
    DoCaloMETAnalysis            = cms.untracked.bool(True),
    DoTcMETAnalysis              = cms.untracked.bool(True),
    DoMuCorrMETAnalysis          = cms.untracked.bool(True),
    DoPfMETAnalysis              = cms.untracked.bool(True),
    DoHTMHTAnalysis              = cms.untracked.bool(True),

    #
    #
    #
    LSBegin = cms.int32(0),
    LSEnd   = cms.int32(-1),                                
                                
    #
    HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
        
    techTrigsAND    = cms.vuint32(0),
    techTrigsOR     = cms.vuint32(40,41),
    techTrigsNOT    = cms.vuint32(36,37,38,39),
    
    #Turn on extra checks
    doPrimaryVertexCheck   = cms.bool(False),
    doHLTPhysicsOn         = cms.bool(False),
    
    #Vertex cleanup parameters
    nvtx_min       = cms.int32(1),
    nvtxtrks_min   = cms.int32(2),
    vtxchi2_max    = cms.double(2.4),
    vtxz_max       = cms.double(20.0),
    
    #Switch on  tight filters for BeamHalo, JetID, HCALnoise
    tightBHFiltering      = cms.bool(True),
    tightHcalFiltering    = cms.bool(True),
    
    #

    #
    # For caloMETAnalysis "met"
    #
    caloMETAnalysis = cms.PSet(
        CaloMETCollectionLabel = cms.InputTag("met"),
        Source = cms.string("CaloMET"),
        FolderName = cms.untracked.string("JetMET/MET/"),
        HLTPathsJetMB = cms.vstring(),
#       When it is empty, it accepts all the triggers
#       HLTPathsJetMB = cms.vstring(
#                     "HLT_L1Jet15","HLT_Jet30","HLT_Jet50","HLT_Jet80","HLT_Jet110","HLT_Jet180",
#                     "HLT_DiJetAve15","HLT_DiJetAve30","HLT_DiJetAve50","HLT_DiJetAve70",
#                     "HLT_DiJetAve130","HLT_DiJetAve220",
#                     "HLT_MinBias"),
        HLT_HighPtJet = cms.string("HLT_Jet50U"),
        HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
        HLT_HighMET   = cms.string("HLT_MET100"),
        HLT_LowMET    = cms.string("HLT_L1MET20"),
        HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
        HLT_Muon      = cms.string("HLT_Mu9"),
        HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
            
        techTrigsAND    = cms.vuint32(0),
        techTrigsOR     = cms.vuint32(40,41),
        techTrigsNOT    = cms.vuint32(36,37,38,39),
        
        #Turn on extra checks
        doPrimaryVertexCheck   = cms.bool(False),
        doHLTPhysicsOn         = cms.bool(False),
        
        #Vertex cleanup parameters
        nvtx_min       = cms.int32(1),
        nvtxtrks_min   = cms.int32(2),
        vtxchi2_max    = cms.double(2.4),
        vtxz_max       = cms.double(20.0),
        
        #Switch on  tight filters for BeamHalo, JetID, HCALnoise
        tightBHFiltering      = cms.bool(True),
        tightJetIDFiltering   = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
        tightHcalFiltering    = cms.bool(True),
        
        #
        
        CaloTowersLabel    = cms.InputTag("towerMaker"),         
        JetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),   # jets used for event cleanup
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
        BeamHaloSummaryLabel = cms.InputTag("BeamHaloSummary"),   # 
        HighPtJetThreshold = cms.double(60.),
        LowPtJetThreshold  = cms.double(15.),
        HighMETThreshold = cms.double(110.),
        LowMETThreshold  = cms.double(30.),
        verbose     = cms.int32(0),
        printOut    = cms.int32(0),
        etThreshold = cms.double(2.),                            # default MET threshold
        allHist     = cms.bool(True),
        allSelection= cms.bool(False)
    ),

    #
    # For caloMETAnalysis "metNoHF"
    #
    caloMETNoHFAnalysis = cms.PSet(
        CaloMETCollectionLabel   = cms.InputTag("metNoHF"),
        Source = cms.string("CaloMETNoHF"),
        FolderName = cms.untracked.string("JetMET/MET/"),
        HLTPathsJetMB = cms.vstring(),
        HLT_HighPtJet = cms.string("HLT_Jet50U"),
        HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
        HLT_HighMET   = cms.string("HLT_MET100"),
        HLT_LowMET    = cms.string("HLT_L1MET20"),
        HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
        HLT_Muon      = cms.string("HLT_Mu9"),
        HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
            
        techTrigsAND    = cms.vuint32(0),
        techTrigsOR     = cms.vuint32(40,41),
        techTrigsNOT    = cms.vuint32(36,37,38,39),
        
        #Turn on extra checks
        doPrimaryVertexCheck   = cms.bool(False),
        doHLTPhysicsOn         = cms.bool(False),
        
        #Vertex cleanup parameters
        nvtx_min       = cms.int32(1),
        nvtxtrks_min   = cms.int32(2),
        vtxchi2_max    = cms.double(2.4),
        vtxz_max       = cms.double(20.0),
        
        #Switch on  tight filters for BeamHalo, JetID, HCALnoise
        tightBHFiltering      = cms.bool(True),
        tightJetIDFiltering   = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
        tightHcalFiltering    = cms.bool(True),
        
        #
        
        CaloTowersLabel    = cms.InputTag("towerMaker"),
        JetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),   # jets used for event cleanup
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
        BeamHaloSummaryLabel    = cms.InputTag("BeamHaloSummary"),   # 
        HighPtJetThreshold = cms.double(60.),
        LowPtJetThreshold  = cms.double(15.),
        HighMETThreshold = cms.double(110.),
        LowMETThreshold  = cms.double(30.),
        verbose     = cms.int32(0),
        printOut    = cms.int32(0),
        etThreshold = cms.double(2.),
        allHist     = cms.bool(False),
        allSelection= cms.bool(False)
    ),

    #
    # For caloMETAnalysis "metHO"
    #
    caloMETHOAnalysis = cms.PSet(
        CaloMETCollectionLabel   = cms.InputTag("metHO"),
        Source = cms.string("CaloMETHO"),
        FolderName = cms.untracked.string("JetMET/MET/"),
        HLTPathsJetMB = cms.vstring(),
        HLT_HighPtJet = cms.string("HLT_Jet50U"),
        HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
        HLT_HighMET   = cms.string("HLT_MET100"),
        HLT_LowMET    = cms.string("HLT_L1MET20"),
        HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
        HLT_Muon      = cms.string("HLT_Mu9"),
        HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
            
        techTrigsAND    = cms.vuint32(0),
        techTrigsOR     = cms.vuint32(40,41),
        techTrigsNOT    = cms.vuint32(36,37,38,39),
        
        #Turn on extra checks
        doPrimaryVertexCheck   = cms.bool(False),
        doHLTPhysicsOn         = cms.bool(False),
        
        #Vertex cleanup parameters
        nvtx_min       = cms.int32(1),
        nvtxtrks_min   = cms.int32(2),
        vtxchi2_max    = cms.double(2.4),
        vtxz_max       = cms.double(20.0),
        
        #Switch on  tight filters for BeamHalo, JetID, HCALnoise
        tightBHFiltering      = cms.bool(True),
        tightJetIDFiltering   = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
        tightHcalFiltering    = cms.bool(True),
        
        #
        
        CaloTowersLabel    = cms.InputTag("towerMaker"),
        JetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),   # jets used for event cleanup
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
        BeamHaloSummaryLabel    = cms.InputTag("BeamHaloSummary"),   # 
        HighPtJetThreshold = cms.double(60.),
        LowPtJetThreshold  = cms.double(15.),
        HighMETThreshold = cms.double(110.),
        LowMETThreshold  = cms.double(30.),
        verbose     = cms.int32(0),
        printOut    = cms.int32(0),
        etThreshold = cms.double(2.),
        allHist     = cms.bool(False),
        allSelection= cms.bool(False)
    ),

    #
    # For caloMETAnalysis
    #
    caloMETNoHFHOAnalysis = cms.PSet(
        CaloMETCollectionLabel   = cms.InputTag("metNoHFHO"),
        Source = cms.string("CaloMETNoHFHO"),
        FolderName = cms.untracked.string("JetMET/MET/"),
        HLTPathsJetMB = cms.vstring(),
        HLT_HighPtJet = cms.string("HLT_Jet50U"),
        HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
        HLT_HighMET   = cms.string("HLT_MET100"),
        HLT_LowMET    = cms.string("HLT_L1MET20"),
        HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
        HLT_Muon      = cms.string("HLT_Mu9"),
        HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
            
        techTrigsAND    = cms.vuint32(0),
        techTrigsOR     = cms.vuint32(40,41),
        techTrigsNOT    = cms.vuint32(36,37,38,39),
        
        #Turn on extra checks
        doPrimaryVertexCheck   = cms.bool(False),
        doHLTPhysicsOn         = cms.bool(False),
        
        #Vertex cleanup parameters
        nvtx_min       = cms.int32(1),
        nvtxtrks_min   = cms.int32(2),
        vtxchi2_max    = cms.double(2.4),
        vtxz_max       = cms.double(20.0),
        
        #Switch on  tight filters for BeamHalo, JetID, HCALnoise
        tightBHFiltering      = cms.bool(True),
        tightJetIDFiltering   = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
        tightHcalFiltering    = cms.bool(True),
        
        #
        
        CaloTowersLabel    = cms.InputTag("towerMaker"),
        JetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),   # jets used for event cleanup
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
        BeamHaloSummaryLabel    = cms.InputTag("BeamHaloSummary"),   # 
        HighPtJetThreshold = cms.double(60.),
        LowPtJetThreshold  = cms.double(15.),
        HighMETThreshold = cms.double(110.),
        LowMETThreshold  = cms.double(30.),
        verbose     = cms.int32(0),
        printOut    = cms.int32(0),
        etThreshold = cms.double(2.),
        allHist     = cms.bool(False),
        allSelection= cms.bool(False)
    ),

                                #
    #
    # For pfMETAnalysis
    #
    pfMETAnalysis = cms.PSet(
        verbose     = cms.int32(0),
        printOut    = cms.int32(0),
        PfMETCollectionLabel         = cms.InputTag("pfMet"),
        PfJetCollectionLabel         = cms.InputTag("iterativeCone5PFJets"),
        Source = cms.string("PfMET"),
        FolderName = cms.untracked.string("JetMET/MET/"),
        HLTPathsJetMB = cms.vstring(),
        HLT_HighPtJet = cms.string("HLT_Jet50U"),
        HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
        HLT_HighMET   = cms.string("HLT_MET100"),
        HLT_LowMET    = cms.string("HLT_L1MET20"),
        HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
        HLT_Muon      = cms.string("HLT_Mu9"),
        HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
            
        techTrigsAND    = cms.vuint32(0),
        techTrigsOR     = cms.vuint32(40,41),
        techTrigsNOT    = cms.vuint32(36,37,38,39),
    
        #Turn on extra checks
        doPrimaryVertexCheck   = cms.bool(False),
        doHLTPhysicsOn         = cms.bool(False),
        
        #Vertex cleanup parameters
        nvtx_min       = cms.int32(1),
        nvtxtrks_min   = cms.int32(2),
        vtxchi2_max    = cms.double(2.4),
        vtxz_max       = cms.double(20.0),
        
        #Switch on  tight filters for BeamHalo, JetID, HCALnoise
        tightBHFiltering      = cms.bool(True),
        tightJetIDFiltering   = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
        tightHcalFiltering    = cms.bool(True),
    
        #
    
        HcalNoiseRBXCollection  = cms.InputTag("hcalnoise"),
        HcalNoiseSummary        = cms.InputTag("hcalnoise"),
        BeamHaloSummaryLabel    = cms.InputTag("BeamHaloSummary"),   # 
        PFCandidates            = cms.InputTag("particleFlow"),
        JetCollectionLabel      = cms.InputTag("iterativeCone5CaloJets"),   # jets used for event cleanup
        JetIDParams = cms.PSet(
            useRecHits = cms.bool(True),
            hbheRecHitsColl = cms.InputTag("hbhereco"),
            hoRecHitsColl   = cms.InputTag("horeco"),
            hfRecHitsColl   = cms.InputTag("hfreco"),
            ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
            eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        ),
        HighPtPFJetThreshold = cms.double(60.),
        LowPtPFJetThreshold  = cms.double(15.),
        HighPFMETThreshold = cms.double(110.),
        LowPFMETThreshold  = cms.double(30.),
        etThreshold = cms.double(2.),
        allHist     = cms.bool(False),
        allSelection= cms.bool(False)
    ),

    #
    # For tcMETAnalysis
    #
    tcMETAnalysis = cms.PSet(
        verbose     = cms.int32(0),
        printOut    = cms.int32(0),
        METCollectionLabel       = cms.InputTag("tcMet"),
        Source = cms.string("TcMET"),
        FolderName = cms.untracked.string("JetMET/MET/"),
        JetCollectionLabel         = cms.InputTag("iterativeCone5CaloJets"),   # jets used for event cleanup
        InputTrackLabel = cms.InputTag("generalTracks"),
        InputMuonLabel = cms.InputTag("muons"),
        InputElectronLabel = cms.InputTag("gsfElectrons"),
        InputBeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        JetIDParams = cms.PSet(
            useRecHits = cms.bool(True),
            hbheRecHitsColl = cms.InputTag("hbhereco"),
            hoRecHitsColl   = cms.InputTag("horeco"),
            hfRecHitsColl   = cms.InputTag("hfreco"),
            ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
            eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        ),
        HLTPathsJetMB = cms.vstring(),
        HLT_HighPtJet = cms.string("HLT_Jet50U"),
        HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
        HLT_HighMET   = cms.string("HLT_MET100"),
        HLT_LowMET    = cms.string("HLT_L1MET20"),
        HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
        HLT_Muon      = cms.string("HLT_Mu9"),
        HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
            
        techTrigsAND    = cms.vuint32(0),
        techTrigsOR     = cms.vuint32(40,41),
        techTrigsNOT    = cms.vuint32(36,37,38,39),
        
        #Turn on extra checks
        doPrimaryVertexCheck   = cms.bool(False),
        doHLTPhysicsOn         = cms.bool(False),
        
        #Vertex cleanup parameters
        nvtx_min       = cms.int32(1),
        nvtxtrks_min   = cms.int32(2),
        vtxchi2_max    = cms.double(2.4),
        vtxz_max       = cms.double(20.0),
        
        #Switch on  tight filters for BeamHalo, JetID, HCALnoise
        tightBHFiltering      = cms.bool(True),
        tightJetIDFiltering   = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
        tightHcalFiltering    = cms.bool(True),
        
        #
        
        HcalNoiseRBXCollection  = cms.InputTag("hcalnoise"),
        HcalNoiseSummary        = cms.InputTag("hcalnoise"),
        BeamHaloSummaryLabel    = cms.InputTag("BeamHaloSummary"),   # 
        HighPtJetThreshold = cms.double(60.),
        LowPtJetThreshold  = cms.double(15.),
        HighMETThreshold = cms.double(110.),
        LowMETThreshold  = cms.double(30.),
        etThreshold = cms.double(2.),
        allHist     = cms.bool(False),
        allSelection= cms.bool(False)
    ),

    #
    # For mucorrMET
    #
    mucorrMETAnalysis = cms.PSet(
        verbose     = cms.int32(0),
        printOut    = cms.int32(0),
        METCollectionLabel         = cms.InputTag("corMetGlobalMuons"),
        Source = cms.string("MuCorrMET"),
        FolderName = cms.untracked.string("JetMET/MET/"),
        HLTPathsJetMB = cms.vstring(),
        HLT_HighPtJet = cms.string("HLT_Jet50U"),
        HLT_LowPtJet  = cms.string("HLT_L1Jet6U"),
        HLT_HighMET   = cms.string("HLT_MET100"),
        HLT_LowMET    = cms.string("HLT_L1MET20"),
        HLT_Ele       = cms.string("HLT_Ele15_LW_L1R"),
        HLT_Muon      = cms.string("HLT_Mu9"),
        HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
            
        techTrigsAND    = cms.vuint32(0),
        techTrigsOR     = cms.vuint32(40,41),
        techTrigsNOT    = cms.vuint32(36,37,38,39),
        
        #Turn on extra checks
        doPrimaryVertexCheck   = cms.bool(False),
        doHLTPhysicsOn         = cms.bool(False),
        
        #Vertex cleanup parameters
        nvtx_min       = cms.int32(1),
        nvtxtrks_min   = cms.int32(2),
        vtxchi2_max    = cms.double(2.4),
        vtxz_max       = cms.double(20.0),
        
        #Switch on  tight filters for BeamHalo, JetID, HCALnoise
        tightBHFiltering      = cms.bool(True),
        tightJetIDFiltering   = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
        tightHcalFiltering    = cms.bool(True),
        
        #
        
        HcalNoiseRBXCollection  = cms.InputTag("hcalnoise"),
        HcalNoiseSummary        = cms.InputTag("hcalnoise"),
        BeamHaloSummaryLabel    = cms.InputTag("BeamHaloSummary"),   # 
        JetCollectionLabel      = cms.InputTag("iterativeCone5CaloJets"),   # jets used for event cleanup
        InputBeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        JetIDParams = cms.PSet(
            useRecHits = cms.bool(True),
            hbheRecHitsColl = cms.InputTag("hbhereco"),
            hoRecHitsColl   = cms.InputTag("horeco"),
            hfRecHitsColl   = cms.InputTag("hfreco"),
            ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
            eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        ),
        HighPtJetThreshold = cms.double(60.),
        LowPtJetThreshold  = cms.double(15.),
        HighMETThreshold = cms.double(110.),
        LowMETThreshold  = cms.double(30.),
        etThreshold = cms.double(2.),
        allHist     = cms.bool(False),
        allSelection= cms.bool(False)
    ),

    #
    # For HTMHTAnalysis
    #
    HTMHTAnalysis = cms.PSet(
        verbose     = cms.int32(0),
        printOut    = cms.int32(0),
        JetCollectionForHTMHTLabel   = cms.InputTag("iterativeCone5CaloJets"),
        FolderName = cms.untracked.string("JetMET/MET/"),
        Source = cms.string("HTMHT"),
        HLTPathsJetMB = cms.vstring(),
        ptThreshold = cms.double(20.)
    ),

    #
    # For jetAnalysis
    #
    jetAnalysis = cms.PSet(
        ptThreshold = cms.double(3.),
        n90HitsMin= cms.int32(-1),
        fHPDMax= cms.double(1.),
        resEMFMin= cms.double(0.),
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
        phiMax  = cms.double(3.2),
        JetIDParams = cms.PSet(
            useRecHits = cms.bool(True),
            hbheRecHitsColl = cms.InputTag("hbhereco"),
            hoRecHitsColl   = cms.InputTag("horeco"),
            hfRecHitsColl   = cms.InputTag("hfreco"),
            ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
            eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        )
    ),

    #
    # For jetcleaning Analysis
    #
    CleanedjetAnalysis = cms.PSet(
        ptThreshold = cms.double(10.),
        n90HitsMin= cms.int32(2),
        fHPDMax= cms.double(0.98),
        resEMFMin= cms.double(0.01),
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
        phiMax  = cms.double(3.2),
        JetIDParams = cms.PSet(
            useRecHits = cms.bool(True),
            hbheRecHitsColl = cms.InputTag("hbhereco"),
            hoRecHitsColl   = cms.InputTag("horeco"),
            hfRecHitsColl   = cms.InputTag("hfreco"),
            ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
            eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        )
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
    # For Cleaned Pt jet Analysis
    #
    CleanedPtAnalysis = cms.PSet(    
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
        n90HitsMin= cms.int32(-1),
        fHPDMax= cms.double(1.),
        resEMFMin= cms.double(0.),
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
        phiMax  = cms.double(3.2),
        JetIDParams = cms.PSet(
            useRecHits = cms.bool(True),
            hbheRecHitsColl = cms.InputTag("hbhereco"),
            hoRecHitsColl   = cms.InputTag("horeco"),
            hfRecHitsColl   = cms.InputTag("hfreco"),
            ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
            eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        )
    ),

    #
    # For Cleaned PF jetAnalysis
    #
    CleanedpfJetAnalysis = cms.PSet(
        ptThreshold = cms.double(10.),
        n90HitsMin= cms.int32(2),
        fHPDMax= cms.double(0.98),
        resEMFMin= cms.double(0.01),
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
        phiMax  = cms.double(3.2),
        JetIDParams = cms.PSet(
            useRecHits = cms.bool(True),
            hbheRecHitsColl = cms.InputTag("hbhereco"),
            hoRecHitsColl   = cms.InputTag("horeco"),
            hfRecHitsColl   = cms.InputTag("hfreco"),
            ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
            eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
        )
    ),

    #
    # For JPT jetAnalysis
    #
    JPTJetAnalysis = jptDQMParameters,

    #
    # For CleanedJPT jetAnalysis
    #
    CleanedJPTJetAnalysis = jptDQMParameters

)
