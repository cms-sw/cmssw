import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetDQMConfig_cff import *      # parameters for all jet analyzers
from DQMOffline.JetMET.jetMETDQMCleanup_cff import *  # parameters for event cleanup

jetDQMAnalyzerAk4CaloUncleaned = cms.EDAnalyzer("JetAnalyzer",
    JetType = cms.string('calo'),#pf, calo or jpt
    JetCorrections = cms.InputTag("dqmAk4CaloL2L3ResidualCorrector"),
    jetsrc = cms.InputTag("ak4CaloJets"),
    l1algoname = cms.string("L1Tech_BPTX_plus_AND_minus.v0"),
    filljetHighLevel =cms.bool(False),
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
    #actually done only for PFJets at the moment
    InputMVAPUIDDiscriminant = cms.InputTag("pileupJetIdProducer","fullDiscriminant"),
    InputCutPUIDDiscriminant = cms.InputTag("pileupJetIdProducer","cutbasedDiscriminant"),
    InputMVAPUIDValue = cms.InputTag("pileupJetIdProducer","fullId"),
    InputCutPUIDValue = cms.InputTag("pileupJetIdProducer","cutbasedId"),
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
    JetCorrections = cms.InputTag("dqmAk4PFL1FastL2L3ResidualCorrector"),
    jetsrc = cms.InputTag("ak4PFJets"),
    #JetCorrections = cms.InputTag("ak4PFCHSL1FastL2L3Corrector"),
    #jetsrc = cms.InputTag("ak4PFJetsCHS"),
    filljetHighLevel  = cms.bool(False),
    DCSFilterForJetMonitoring = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
      #DebugOn = cms.untracked.bool(True),
      alwaysPass = cms.untracked.bool(False)
    )
)


jetDQMAnalyzerAk4PFCleaned=jetDQMAnalyzerAk4PFUncleaned.clone(
    JetCleaningFlag = cms.untracked.bool(True),
    filljetHighLevel = cms.bool(False),
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = cms.double(20.),
        asymmetryThirdJetCut = cms.double(30),
        balanceThirdJetCut = cms.double(0.2),
        )
)

jetDQMAnalyzerAk4PFCHSCleaned=jetDQMAnalyzerAk4PFCleaned.clone(
    filljetHighLevel =cms.bool(True),
    JetCorrections = cms.InputTag("dqmAk4PFCHSL1FastL2L3ResidualCorrector"),
    jetsrc = cms.InputTag("ak4PFJetsCHS"),
    #actually done only for PFJets at the moment
    InputMVAPUIDDiscriminant = cms.InputTag("pileupJetIdProducerChs","fullDiscriminant"),
    InputCutPUIDDiscriminant = cms.InputTag("pileupJetIdProducerChs","cutbasedDiscriminant"),
    InputMVAPUIDValue = cms.InputTag("pileupJetIdProducerChs","fullId"),
    InputCutPUIDValue = cms.InputTag("pileupJetIdProducerChs","cutbasedId"),
)

jetDQMAnalyzerAk4PFCHSUncleanedMiniAOD=jetDQMAnalyzerAk4PFUncleaned.clone(
    filljetHighLevel =cms.bool(True),
    CleaningParameters = cleaningParameters.clone(
        vertexCollection    = cms.InputTag( "goodOfflinePrimaryVerticesDQMforMiniAOD" ),
        ),
    JetType = cms.string('miniaod'),#pf, calo or jpt
    jetsrc = cms.InputTag("slimmedJets"),
)

jetDQMAnalyzerAk4PFCHSCleanedMiniAOD=jetDQMAnalyzerAk4PFCleaned.clone(
    CleaningParameters = cleaningParameters.clone(
        vertexCollection    = cms.InputTag( "goodOfflinePrimaryVerticesDQMforMiniAOD" ),
        ),
    JetType = cms.string('miniaod'),#pf, calo or jpt
    jetsrc = cms.InputTag("slimmedJets"),
)

jetDQMAnalyzerIC5CaloHIUncleaned=jetDQMAnalyzerAk4CaloUncleaned.clone(
    filljetHighLevel =cms.bool(True),
    CleaningParameters = cleaningParameters.clone(
        bypassAllPVChecks  = cms.bool(False),
        vertexCollection = cms.InputTag( "hiSelectedVertex" ),
        ),
    JetType = cms.string('calo'),#pf, calo or jpt
    JetCorrections = cms.InputTag(""),# no jet correction available yet?
    jetsrc = cms.InputTag("iterativeConePu5CaloJets"),
    JetCleaningFlag            = cms.untracked.bool(False),  
    runcosmics                 = cms.untracked.bool(True),   
    DCSFilterForJetMonitoring = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        #DebugOn = cms.untracked.bool(True),
        alwaysPass = cms.untracked.bool(False)
    )
)

jetDQMAnalyzerAkVs3PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs3PFJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)
jetDQMAnalyzerAkPU3Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Pu'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akPu3CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkPU4Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Pu'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akPu4CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

jetDQMAnalyzerAkPU5Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Pu'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akPu5CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkPU3PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Pu'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akPu3PFJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkPU4PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Pu'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akPu4PFJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkPU5PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Pu'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akPu5PFJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs2Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs2CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs3Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs3CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs4Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs4CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs5Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs5CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundCalo"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)
'''
jetDQMAnalyzerAkVs6Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs6CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs7Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                      JetType = cms.untracked.string('calo'),
                                      UEAlgo = cms.untracked.string('Vs'),
                                      OutputFile = cms.untracked.string(''),
                                      src = cms.InputTag("akVs7CaloJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                      JetCorrections = cms.string(""),
                                      recoJetPtThreshold = cms.double(10),        
                                      RThreshold = cms.double(0.3),
                                      reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs2PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs2PFJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)
'''

jetDQMAnalyzerAkVs3PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs3PFJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    #srcRho = cms.InputTag("akVs3PFJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)


jetDQMAnalyzerAkVs4PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs4PFJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                    #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs5PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs5PFJets"),
                                    PFcands = cms.InputTag("particleFlowTmp"),
                                    Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                    centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)
'''
jetDQMAnalyzerAkVs6PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs6PFJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs7PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                    JetType = cms.untracked.string('pf'),
                                    UEAlgo = cms.untracked.string('Vs'),
                                    OutputFile = cms.untracked.string(''),
                                    src = cms.InputTag("akVs7PFJets"),
                                      PFcands = cms.InputTag("particleFlowTmp"),
                                      Background = cms.InputTag("voronoiBackgroundPF"),
                                      #srcRho = cms.InputTag("iterativeConePu5CaloJets","rho"),
                                      centrality = cms.InputTag("hiCentrality"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(10),        
                                    RThreshold = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
)				    			    
'''
