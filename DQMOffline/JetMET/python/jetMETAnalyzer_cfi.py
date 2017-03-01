import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jptDQMConfig_cff import *      # parameters for jpt analyzer
from DQMOffline.JetMET.jetDQMConfig_cff import *      # parameters for all jet analyzers
from DQMOffline.JetMET.metDQMConfig_cff import *      # parameters for all met analyzers
from DQMOffline.JetMET.jetMETDQMCleanup_cff import *  # parameters for event cleanup

jetMETAnalyzer = cms.EDAnalyzer("JetMETAnalyzer",

    #
    # Output files
    #
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('jetMETMonitoring.root'),

    #
    #
    #
    highPtJetTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_highptjet' ),
        hltPaths       = cms.vstring( 'HLT_Jet300_v','HLT_Jet300_v6','HLT_Jet300_v7','HLT_Jet300_v8' ),
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    lowPtJetTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_lowptjet' ),
        hltPaths       = cms.vstring( 'HLT_Jet60_v','HLT_Jet60_v6','HLT_Jet60_v7','HLT_Jet60_v8' ),
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),

    TriggerResultsLabel        = cms.InputTag("TriggerResults::HLT"),
    processname                = cms.string("HLT"),

    #
    # Jet-related
    #
    DoPFJetAnalysis            = cms.untracked.bool(False),
    DoPFJetCleaning            = cms.untracked.bool(True),

    DoJPTJetAnalysis           = cms.untracked.bool(False),
    DoJPTJetCleaning           = cms.untracked.bool(True),

    DoJetAnalysis              = cms.untracked.bool(True),
    DoJetCleaning              = cms.untracked.bool(True),
    DoIterativeCone            = cms.untracked.bool(False),
    DoSisCone                  = cms.untracked.bool(False),

    DoJetPtAnalysis            = cms.untracked.bool(False),
    DoJetPtCleaning            = cms.untracked.bool(False),
    DoDiJetSelection           = cms.untracked.bool(True),

    PFJetsCollectionLabel      = cms.InputTag("ak4PFJets"),
    JPTJetsCollectionLabel     = cms.InputTag("JetPlusTrackZSPCorJetAntiKt4"),
    SCJetsCollectionLabel      = cms.InputTag("sisCone5CaloJets"),
    AKJetsCollectionLabel      = cms.InputTag("ak4CaloJets"),
    ICJetsCollectionLabel      = cms.InputTag("iterativeCone5CaloJets"),

    #
    # MET-related
    #
    DoCaloMETAnalysis            = cms.untracked.bool(True),
    DoTcMETAnalysis              = cms.untracked.bool(True),
    DoMuCorrMETAnalysis          = cms.untracked.bool(False),
    DoPfMETAnalysis              = cms.untracked.bool(True),
    DoHTMHTAnalysis              = cms.untracked.bool(False),

    #
    #
    #
    LSBegin = cms.int32(0),
    LSEnd   = cms.int32(-1),

    #Cleanup parameters
    CleaningParameters = cleaningParameters.clone(),

    #
    # For caloMETAnalysis "met"
    #
     caloMETAnalysis = metDQMParameters.clone(
         allHist = cms.bool(True)
     ),

     #
     # For caloMETAnalysis "metNoHF"
     #
     caloMETNoHFAnalysis = metDQMParameters.clone(
         METCollectionLabel = cms.InputTag("metNoHF"),
         Source             = cms.string("CaloMETNoHF")
     ),

     #
     # For caloMETAnalysis "metHO"
     #
     caloMETHOAnalysis = metDQMParameters.clone(
         METCollectionLabel = cms.InputTag("metHO"),
         Source             = cms.string("CaloMETHO"),
         DetectorTypes = cms.untracked.string("ecal:hbhe:hf:ho"),
         #DebugOn = cms.untracked.bool(True),
         Filter = cms.untracked.bool(True)
     ),

     #
     # For caloMETAnalysis
     #
     caloMETNoHFHOAnalysis = metDQMParameters.clone(
         METCollectionLabel = cms.InputTag("metNoHFHO"),
         Source             = cms.string("CaloMETNoHFHO"),
         DCSFilter = cms.PSet(
           DetectorTypes = cms.untracked.string("ecal:hbhe:hf:ho"),
           #DebugOn = cms.untracked.bool(True),
           Filter = cms.untracked.bool(True)
         )
     ),

     #
     # For pfMETAnalysis
     #
     pfMETAnalysis = metDQMParameters.clone(
         METCollectionLabel   = cms.InputTag("pfMet"),
         Source               = cms.string("PfMET"),
         PfJetCollectionLabel = cms.InputTag("iterativeCone5PFJets"),
         PFCandidates         = cms.InputTag("particleFlow"),
         DCSFilter = cms.PSet(
           DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
           #DebugOn = cms.untracked.bool(True),
           Filter = cms.untracked.bool(True)
         )
     ),

     #
     # For tcMETAnalysis
     #
     tcMETAnalysis = metDQMParameters.clone(
         METCollectionLabel = cms.InputTag("tcMet"),
         Source             = cms.string("TcMET"),
         InputTrackLabel    = cms.InputTag("generalTracks"),
         InputMuonLabel     = cms.InputTag("muons"),
         InputElectronLabel = cms.InputTag("gedGsfElectrons"),
         InputBeamSpotLabel = cms.InputTag("offlineBeamSpot"),
         DCSFilter = cms.PSet(
           DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
           #DebugOn = cms.untracked.bool(True),
           Filter = cms.untracked.bool(True)
         )
     ),

     #
     # For mucorrMET
     #
     mucorrMETAnalysis = metDQMParameters.clone(
         METCollectionLabel = cms.InputTag("caloMetM"),
         Source             = cms.string("MuCorrMET"),
         InputBeamSpotLabel = cms.InputTag("offlineBeamSpot"),
         DCSFilter = cms.PSet(
           DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:muon"),
           #DebugOn = cms.untracked.bool(True),
           Filter = cms.untracked.bool(True)
         )
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
    jetAnalysis = jetDQMParameters.clone(),

    #
    # For jetcleaning Analysis
    #
    CleanedjetAnalysis = cleanedJetDQMParameters.clone(),

    #
    # For dijet Analysis
    #
    DijetAnalysis = cleanedJetDQMParameters.clone(
        makedijetselection = cms.int32(1),
        ptThreshold = cms.double(20.),
        fillJIDPassFrac   = cms.int32(1)
    ),

    PFDijetAnalysis = cleanedJetDQMParameters.clone(
        makedijetselection = cms.int32(1),
        ptThreshold = cms.double(20.),
        fillJIDPassFrac   = cms.int32(1)
    ),

    #
    # For Pt jet Analysis
    #
    PtAnalysis = jetDQMParameters.clone(),

    #
    # For Cleaned Pt jet Analysis
    #
    CleanedPtAnalysis = cleanedJetDQMParameters.clone(),

    #
    # For PF jetAnalysis
    #
    pfJetAnalysis = jetDQMParameters.clone(
    ptThreshold = cms.double(20.),
    TightCHFMin = cms.double(0.0),
    TightNHFMax = cms.double(1.0),
    TightCEFMax = cms.double(1.0),
    TightNEFMax = cms.double(1.0),
    LooseCHFMin = cms.double(0.0),
    LooseNHFMax = cms.double(0.9),
    LooseCEFMax = cms.double(1.0),
    LooseNEFMax = cms.double(0.9),
    ThisCHFMin = cms.double(-999.),
    ThisNHFMax = cms.double(999.),
    ThisCEFMax = cms.double(999.),
    ThisNEFMax = cms.double(999.)
    ),

    #
    # For Cleaned PF jetAnalysis
    #
    CleanedpfJetAnalysis = cleanedJetDQMParameters.clone(
    ptThreshold = cms.double(20.)
    ),

    #
    # For JPT jetAnalysis
    #
    JPTJetAnalysis = jptDQMParameters.clone(
    ),

    #
    # For CleanedJPT jetAnalysis
    #
    CleanedJPTJetAnalysis = cleanedjptDQMParameters.clone(
    ),

    #
    # DCS
    #
    DCSFilterCalo = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    ),
    DCSFilterPF = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    ),
    DCSFilterJPT = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    ),
    DCSFilterAll = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf:ho:pixel:sistrip:es:muon"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    )

)
