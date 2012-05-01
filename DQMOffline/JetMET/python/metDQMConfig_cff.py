import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETDQMCleanup_cff import *

metDQMParameters = cms.PSet(

    METCollectionLabel     = cms.InputTag("met"),
    
    Source = cms.string("CaloMET"),
    FolderName = cms.untracked.string("JetMET/MET/"),

    CleaningParameters = cleaningParameters.clone(),

    HLTPathsJetMB = cms.vstring(),
#    When it is empty, it accepts all the triggers
#    HLTPathsJetMB = cms.vstring(
#        "HLT_L1Jet15","HLT_Jet30","HLT_Jet50","HLT_Jet80","HLT_Jet110","HLT_Jet180",
#        "HLT_DiJetAve15","HLT_DiJetAve30","HLT_DiJetAve50","HLT_DiJetAve70",
#        "HLT_DiJetAve130","HLT_DiJetAve220",
#        "HLT_MinBias"),

    highPtJetTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_highptjet' ),
        hltPaths       = cms.vstring( 'HLT_Jet300_v1',
                                      'HLT_Jet300_v2',
                                      'HLT_Jet300_v3',
                                      'HLT_Jet300_v4',
                                      'HLT_Jet300_v5',
                                      'HLT_Jet300_v6',
                                      'HLT_Jet300_v7' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    lowPtJetTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_lowptjet' ),
        hltPaths       = cms.vstring( 'HLT_Jet80_v1',
                                      'HLT_Jet80_v2',
                                      'HLT_Jet80_v3',
                                      'HLT_Jet80_v4',
                                      'HLT_Jet80_v5',
                                      'HLT_Jet80_v6',
                                      'HLT_Jet80_v7',
                                      'HLT_Jet80_v8',
                                      'HLT_Jet80_v9' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    minBiasTrigger = cms.PSet(
        #andOr         = cms.bool( False ),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_minbias' ),
        hltPaths       = cms.vstring( 'HLT_L1Tech_BSC_minBias_threshold1_v1',
                                      'HLT_L1Tech_BSC_minBias_threshold1_v2',
                                      'HLT_L1Tech_BSC_minBias_threshold1_v3',
                                      'HLT_L1Tech_BSC_minBias_threshold1_v4',
                                      'HLT_L1Tech_BSC_minBias_threshold1_v5' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    highMETTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_highmet' ),
        hltPaths       = cms.vstring( 'HLT_MET200_v1',
                                      'HLT_MET200_v2',
                                      'HLT_MET200_v3',
                                      'HLT_MET200_v4',
                                      'HLT_MET200_v5',
                                      'HLT_MET200_v6',
                                      'HLT_MET200_v7' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    lowMETTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_lowmet' ),
        hltPaths       = cms.vstring( 'HLT_MET120_v1',
                                      'HLT_MET120_v2',
                                      'HLT_MET120_v3',
                                      'HLT_MET120_v4',
                                      'HLT_MET120_v5',
                                      'HLT_MET120_v6',
                                      'HLT_MET120_v7' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    eleTrigger = cms.PSet(
        #andOr         = cms.bool( False ),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_ele' ),
        hltPaths       = cms.vstring( 'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v1',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v2',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v3',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v4',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v5',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v6',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v7',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v8',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v9',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v10',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v11',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v12',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v13',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v14',
                                      'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v15' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    muonTrigger = cms.PSet(
        #andOr         = cms.bool( False ),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_muon' ),
        hltPaths       = cms.vstring( 'HLT_IsoMu17_v1',
                                      'HLT_IsoMu17_v2',
                                      'HLT_IsoMu17_v3',
                                      'HLT_IsoMu17_v4',
                                      'HLT_IsoMu17_v5',
                                      'HLT_IsoMu17_v6',
                                      'HLT_IsoMu17_v7',
                                      'HLT_IsoMu17_v8',
                                      'HLT_IsoMu17_v9',
                                      'HLT_IsoMu17_v10',
                                      'HLT_IsoMu17_v11',
                                      'HLT_IsoMu17_v12',
                                      'HLT_IsoMu17_v13',
                                      'HLT_IsoMu17_v14',
                                      'HLT_IsoMu17_v15' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),

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
    BeamHaloSummaryLabel    = cms.InputTag("BeamHaloSummary"),    

    HighPtJetThreshold = cms.double(60.),
    LowPtJetThreshold  = cms.double(15.),
    HighMETThreshold   = cms.double(110.),
    LowMETThreshold    = cms.double(30.),

    verbose     = cms.int32(0),
    printOut    = cms.int32(0),

    etThreshold  = cms.double(2.),
    allHist      = cms.bool(False),
    allSelection = cms.bool(False),
    cleanupSelection = cms.bool(True),
    
    #Parameters set only for PFMETAnalyzer
    PfJetCollectionLabel   = cms.InputTag(""),
    PFCandidates       = cms.InputTag(""),
    
    #Parameters set for METAnalyzer
    InputBeamSpotLabel = cms.InputTag(""),
    InputTrackLabel    = cms.InputTag(""),
    InputMuonLabel     = cms.InputTag(""),
    InputElectronLabel = cms.InputTag(""),

    DCSFilter = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    )

)
