import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetDQMConfig_cff import *      # parameters for all jet analyzers
from DQMOffline.JetMET.jetMETDQMCleanup_cff import *  # parameters for event cleanup


from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4CaloL2L3,ak4CaloL2Relative,ak4CaloL3Absolute

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak7CaloL2L3,ak7CaloL2Relative,ak7CaloL3Absolute

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFL1FastL2L3,ak4PFL1Fastjet,ak4PFL2Relative,ak4PFL3Absolute

#from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4JPTL1FastL2L3,ak4JPTL1Fastjet,ak4JPTL2Relative,ak4JPTL3Absolute
#newAk4JPTL1FastL2L3 = ak4JPTL1FastL2L3.clone()

jetDQMAnalyzerAk4CaloUncleaned = cms.EDAnalyzer("JetAnalyzer",
    OutputMEsInRootFile = cms.bool(False),
    OutputFile = cms.string('jetMETMonitoring.root'),
    JetType = cms.string('calo'),#pf, calo or jpt
    JetCorrections = cms.string("ak4CaloL2L3"),
    jetsrc = cms.InputTag("ak4CaloJets"),
    filljetHighLevel =cms.bool(True),
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
    InputJetIDValueMap         = cms.InputTag("ak4JetID"), 
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

jetDQMAnalyzerAk4CaloCleaned=jetDQMAnalyzerAk4CaloUncleaned.clone(
    JetCleaningFlag   = cms.untracked.bool(True),
    filljetHighLevel  = cms.bool(False),
    CleaningParameters = cleaningParameters.clone(
        bypassAllPVChecks = cms.bool(False),
    ),
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = cms.double(20.),
        asymmetryThirdJetCut = cms.double(30),
        balanceThirdJetCut   = cms.double(0.2), 
       )  
)


#jetDQMAnalyzerAk4JPTCleaned=jetDQMAnalyzerAk4CaloCleaned.clone(
#    JetType = cms.string('jpt'),#pf, calo or jpt
#    JetCorrections = cms.string("newAk4JPTL1FastL2L3"),
#    jetsrc = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
#    JetCleaningFlag   = cms.untracked.bool(True),
#    filljetHighLevel  = cms.bool(False),
#    DCSFilterForJetMonitoring = cms.PSet(
#        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
#        #DebugOn = cms.untracked.bool(True),
#        alwaysPass = cms.untracked.bool(False)
#        )
#)

jetDQMAnalyzerAk4PFUncleaned=jetDQMAnalyzerAk4CaloUncleaned.clone(
    CleaningParameters = cleaningParameters.clone(
       bypassAllPVChecks  = cms.bool(False),
    ),
    #for PFJets: LOOSE,TIGHT
    JetIDQuality               = cms.string("LOOSE"),
    #options for Calo and JPT: PURE09,DQM09,CRAFT08
    #for PFJets: FIRSTDATA
    JetIDVersion               = cms.string("FIRSTDATA"),
    JetType = cms.string('pf'),#pf, calo or jpt
    JetCorrections = cms.string("ak4PFL1FastL2L3"),
    jetsrc = cms.InputTag("ak4PFJets"),
    filljetHighLevel  = cms.bool(False),
    DCSFilterForJetMonitoring = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
      #DebugOn = cms.untracked.bool(True),
      alwaysPass = cms.untracked.bool(False)
    )
)

jetDQMAnalyzerAk4PFCleaned=jetDQMAnalyzerAk4PFUncleaned.clone(
    JetCleaningFlag   = cms.untracked.bool(True),
    filljetHighLevel  = cms.bool(False),
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = cms.double(20.),
        asymmetryThirdJetCut = cms.double(30),
        balanceThirdJetCut   = cms.double(0.2),
        )
)

jetDQMAnalyzerIC5CaloHIUncleaned=jetDQMAnalyzerAk4CaloUncleaned.clone(
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
