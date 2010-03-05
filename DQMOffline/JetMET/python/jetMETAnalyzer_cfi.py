import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jptDQMConfig_cff import *     #parameters for jpt analyzer
from DQMOffline.JetMET.jetDQMConfig_cff import *     #parameters for all jet analyzers
from DQMOffline.JetMET.metDQMConfig_cff import *     #parameters for all met analyzers
from DQMOffline.JetMET.jetMETDQMCleanup_cff import * #parameters for event cleanup

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
    DoPFJetAnalysis            = cms.untracked.bool(True),#True
    DoPFJetCleaning            = cms.untracked.bool(True),#True

    DoJPTJetAnalysis           = cms.untracked.bool(True),#True
    DoJPTJetCleaning           = cms.untracked.bool(True),#True

    DoJetAnalysis              = cms.untracked.bool(True),
    DoJetCleaning              = cms.untracked.bool(True),
    DoIterativeCone            = cms.untracked.bool(False),
    DoSisCone            = cms.untracked.bool(False),                               

    DoJetPtAnalysis            = cms.untracked.bool(False),                           
    DoJetPtCleaning            = cms.untracked.bool(False),                           
    DoDiJetSelection           = cms.untracked.bool(True),

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
         Source             = cms.string("CaloMETHO")
     ),

     #
     # For caloMETAnalysis
     #
     caloMETNoHFHOAnalysis = metDQMParameters.clone(
         METCollectionLabel = cms.InputTag("metNoHFHO"),
         Source             = cms.string("CaloMETNoHFHO")
     ),
 
     #
     # For pfMETAnalysis
     #
     pfMETAnalysis = metDQMParameters.clone(
         METCollectionLabel   = cms.InputTag("pfMet"),
         Source               = cms.string("PfMET"),
         PfJetCollectionLabel = cms.InputTag("iterativeCone5PFJets"),
         PFCandidates         = cms.InputTag("particleFlow")
     ),
 
     #
     # For tcMETAnalysis
     #
     tcMETAnalysis = metDQMParameters.clone(
         METCollectionLabel     = cms.InputTag("tcMet"),
         Source                 = cms.string("TcMET"),
         InputTrackLabel    = cms.InputTag("generalTracks"),
         InputMuonLabel     = cms.InputTag("muons"),
         InputElectronLabel = cms.InputTag("gsfElectrons"),
         InputBeamSpotLabel = cms.InputTag("offlineBeamSpot")
     ),

     #
     # For mucorrMET
     #
     mucorrMETAnalysis = metDQMParameters.clone(
         METCollectionLabel = cms.InputTag("corMetGlobalMuons"),
         Source             = cms.string("MuCorrMET"),
         InputBeamSpotLabel = cms.InputTag("offlineBeamSpot")
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
        ptThreshold = cms.double(8.),
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
    TightCHFMin = cms.double(0.0),
    TightNHFMax = cms.double(1.0),
    TightCEFMax = cms.double(1.0),
    TightNEFMax = cms.double(1.0),
    LooseCHFMin = cms.double(0.0),
    LooseNHFMax = cms.double(0.9),
    LooseCEFMax = cms.double(1.0),
    LooseNEFMax = cms.double(0.9),
    ),

    #
    # For Cleaned PF jetAnalysis
    #
    CleanedpfJetAnalysis = cleanedJetDQMParameters.clone(),

    #
    # For JPT jetAnalysis
    #
    JPTJetAnalysis = jptDQMParameters.clone(
    ),

    #
    # For CleanedJPT jetAnalysis
    #
    CleanedJPTJetAnalysis = jptDQMParameters.clone(
    )

)
