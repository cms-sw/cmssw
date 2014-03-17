import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetDQMConfig_cff import *      # parameters for all jet analyzers
from DQMOffline.JetMET.jetMETDQMCleanup_cff import *  # parameters for event cleanup


from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5CaloL2L3,ak5CaloL2Relative,ak5CaloL3Absolute
newAk5CaloL2L3 = ak5CaloL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak7CaloL2L3,ak7CaloL2Relative,ak7CaloL3Absolute
newAk7CaloL2L3 = ak7CaloL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5PFL1FastL2L3,ak5PFL1Fastjet,ak5PFL2Relative,ak5PFL3Absolute
newAk5PFL1FastL2L3 = ak5PFL1FastL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5JPTL1FastL2L3,ak5JPTL1Fastjet,ak5JPTL2Relative,ak5JPTL3Absolute
newAk5JPTL1FastL2L3 = ak5JPTL1FastL2L3.clone()

jetDQMAnalyzerAk5CaloUncleaned = cms.EDAnalyzer("JetAnalyzer",
    OutputMEsInRootFile = cms.bool(False),
    OutputFile = cms.string('jetMETMonitoring.root'),
    JetType = cms.string('calo'),#pf, calo or jpt
    JetCorrections = cms.string("newAk5CaloL2L3"),
    jetsrc = cms.InputTag("ak5CaloJets"),
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

    JetCleaningFlag            = cms.untracked.bool(False),       

    runcosmics                 = cms.untracked.bool(False),                
                                
    #Cleanup parameters
    CleaningParameters = cleaningParameters.clone(
        bypassAllPVChecks = cms.bool(True),
        ),

    #for JPT and CaloJetID  
    InputJetIDValueMap         = cms.InputTag("ak5JetID"), 
    #options for Calo and JPT: LOOSE,LOOSE_AOD,TIGHT,MINIMAL
    #for PFJets: LOOSE,TIGHT
    JetIDQuality               = cms.string("LOOSE"),
    #options for Calo and JPT: PURE09,DQM09,CRAFT08
    #for PFJets: FIRSTDATA
    JetIDVersion               = cms.string("PURE09"),
    #
    # For jetAnalysis
    #
    jetAnalysis = jetDQMParameters.clone(),

    #
    # DCS
    #                             
    DCSFilterForJetMonitoring = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf"),
      #DebugOn = cms.untracked.bool(True),
      alwaysPass = cms.untracked.bool(False)
    )
)

jetDQMAnalyzerAk5CaloCleaned=jetDQMAnalyzerAk5CaloUncleaned.clone(
    JetCleaningFlag   = cms.untracked.bool(True),
    CleaningParameters = cleaningParameters.clone(
        bypassAllPVChecks = cms.bool(False),
    ),
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = cms.double(20.),
        asymmetryThirdJetCut = cms.double(30),
        balanceThirdJetCut   = cms.double(0.2), 
       )  
)


jetDQMAnalyzerAk5JPTCleaned=jetDQMAnalyzerAk5CaloCleaned.clone(
    JetType = cms.string('jpt'),#pf, calo or jpt
    JetCorrections = cms.string("newAk5JPTL1FastL2L3"),
    jetsrc = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    JetCleaningFlag   = cms.untracked.bool(True),
    DCSFilterForJetMonitoring = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        #DebugOn = cms.untracked.bool(True),
        alwaysPass = cms.untracked.bool(False)
        )
)

jetDQMAnalyzerAk5PFUncleaned=jetDQMAnalyzerAk5CaloUncleaned.clone(
    CleaningParameters = cleaningParameters.clone(
       bypassAllPVChecks  = cms.bool(False),
    ),
    #for PFJets: LOOSE,TIGHT
    JetIDQuality               = cms.string("LOOSE"),
    #options for Calo and JPT: PURE09,DQM09,CRAFT08
    #for PFJets: FIRSTDATA
    JetIDVersion               = cms.string("FIRSTDATA"),
    JetType = cms.string('pf'),#pf, calo or jpt
    JetCorrections = cms.string("newAk5PFL1FastL2L3"),
    jetsrc = cms.InputTag("ak5PFJets"),
    DCSFilterForJetMonitoring = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
      #DebugOn = cms.untracked.bool(True),
      alwaysPass = cms.untracked.bool(False)
    )
)

jetDQMAnalyzerAk5PFCleaned=jetDQMAnalyzerAk5PFUncleaned.clone(
    JetCleaningFlag   = cms.untracked.bool(True),
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = cms.double(20.),
        asymmetryThirdJetCut = cms.double(30),
        balanceThirdJetCut   = cms.double(0.2),
        )
)

jetDQMAnalyzerIC5CaloHIUncleaned=jetDQMAnalyzerAk5CaloUncleaned.clone(
    CleaningParameters = cleaningParameters.clone(
        bypassAllPVChecks  = cms.bool(False),
        vertexCollection = cms.InputTag( "hiSelectedVertex" ),
        ),
    JetType = cms.string('calo'),#pf, calo or jpt
    JetCorrections = cms.string(""),# no jet correction available yet?
    jetsrc = cms.InputTag("iterativeConePu5CaloJets"),
    JetCleaningFlag            = cms.untracked.bool(False),   
    runcosmics                 = cms.untracked.bool(True),   
    DCSFilterForJetMonitoring = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        #DebugOn = cms.untracked.bool(True),
        alwaysPass = cms.untracked.bool(False)
    )
)
