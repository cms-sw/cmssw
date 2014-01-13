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

jetAnalyzerAk5CaloUncleaned = cms.EDAnalyzer("JetAnalyzer",
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
            
    DoDiJetSelection           = cms.untracked.bool(False),

    JetCleaningFlag            = cms.untracked.bool(False),

    LSBegin = cms.int32(0),
    LSEnd   = cms.int32(-1),                                
                                
    #Cleanup parameters
    CleaningParameters = cleaningParameters.clone(
        doPrimaryVertexCheck = cms.bool(False),
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

    fillJIDPassFrac   = cms.int32(1),

    #
    # DCS
    #                             
    DCSFilterForJetMonitoring = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf"),
      #DebugOn = cms.untracked.bool(True),
      alwaysPass = cms.untracked.bool(False)
    )
)

jetAnalyzerAk5CaloCleaned=jetAnalyzerAk5CaloUncleaned.clone(
    fillJIDPassFrac   = cms.int32(0),
    JetCleaningFlag   = cms.untracked.bool(True),
    CleaningParameters = cleaningParameters.clone(
        doPrimaryVertexCheck = cms.bool(True),
    ),
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = cms.double(20.),
        asymmetryThirdJetCut = cms.double(30),
        balanceThirdJetCut   = cms.double(0.2),
        n90HitsMin  = cms.int32(2),
        fHPDMax     = cms.double(0.98),
        resEMFMin   = cms.double(0.01),  
       )  
)

jetAnalyzerAk5CaloDiJetCleaned=jetAnalyzerAk5CaloCleaned.clone(
        DoDiJetSelection=cms.untracked.bool(True),
)

jetAnalyzerAk5JPTCleaned=jetAnalyzerAk5CaloCleaned.clone(
    JetType = cms.string('jpt'),#pf, calo or jpt
    JetCorrections = cms.string("newAk5JPTL1FastL2L3"),
    jetsrc = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    fillJIDPassFrac   = cms.int32(0),
    JetCleaningFlag   = cms.untracked.bool(True)
)
jetAnalyzerAk5JPTCleaned.DCSFilterForJetMonitoring.DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon")

jetAnalyzerAk5PFUncleaned=jetAnalyzerAk5CaloUncleaned.clone(
    CleaningParameters = cleaningParameters.clone(
        doPrimaryVertexCheck = cms.bool(True),
    ),
    #for PFJets: LOOSE,TIGHT
    JetIDQuality               = cms.string("LOOSE"),
    #options for Calo and JPT: PURE09,DQM09,CRAFT08
    #for PFJets: FIRSTDATA
    JetIDVersion               = cms.string("FIRSTDATA"),
    JetType = cms.string('pf'),#pf, calo or jpt
    JetCorrections = cms.string("newAk5PFL1FastL2L3"),
    jetsrc = cms.InputTag("ak5PFJets")
)
jetAnalyzerAk5PFUncleaned.DCSFilterForJetMonitoring.DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon")

jetAnalyzerAk5PFCleaned=jetAnalyzerAk5PFUncleaned.clone(
    fillJIDPassFrac   = cms.int32(0),
    JetCleaningFlag   = cms.untracked.bool(True),
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = cms.double(20.),
        asymmetryThirdJetCut = cms.double(30),
        balanceThirdJetCut   = cms.double(0.2),
        ThisCHFMin = cms.double(0.0),
        ThisNHFMax = cms.double(1.0),
        ThisCEFMax = cms.double(1.0),
        ThisNEFMax = cms.double(1.0),
        )
)

jetAnalyzerAk5PFDiJetCleaned=jetAnalyzerAk5PFCleaned.clone(
    DoDiJetSelection=cms.untracked.bool(True),
    )
