import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetDQMConfig_cff import *      # parameters for all jet analyzers
from DQMOffline.JetMET.jetMETDQMCleanup_cff import *  # parameters for event cleanup

jetDQMAnalyzerAk4CaloUncleaned = cms.EDAnalyzer("JetAnalyzer",
    JetType = cms.string('calo'),#pf, calo or jpt
    JetCorrections = cms.InputTag("dqmAk4CaloL2L3ResidualCorrector"),
    jetsrc = cms.InputTag("ak4CaloJets"),
    METCollectionLabel     = cms.InputTag("caloMet"),
    muonsrc = cms.InputTag("muons"),
    l1algoname = cms.string("L1Tech_BPTX_plus_AND_minus.v0"),
    filljetHighLevel =cms.bool(False),
    fillsubstructure =cms.bool(False),
    ptMinBoosted = cms.double(400.),
    #
    #
    #
    highPtJetTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltPaths       = cms.vstring( 'HLT_PFJet450_v*'), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    lowPtJetTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltPaths       = cms.vstring( 'HLT_PFJet80_v*'), 
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
    InputMVAPUIDDiscriminant = cms.InputTag("pileupJetIdEvaluatorDQM","fullDiscriminant"),
    InputCutPUIDDiscriminant = cms.InputTag("pileupJetIdEvaluatorDQM","cutbasedDiscriminant"),
    InputMVAPUIDValue = cms.InputTag("pileupJetIdEvaluatorDQM","fullId"),
    InputCutPUIDValue = cms.InputTag("pileupJetIdEvaluatorDQM","cutbasedId"),

    InputQGMultiplicity = cms.InputTag("QGTagger", "mult"),
    InputQGLikelihood = cms.InputTag("QGTagger", "qgLikelihood"),
    InputQGPtDToken = cms.InputTag("QGTagger", "ptD"),
    InputQGAxis2 = cms.InputTag("QGTagger", "axis2"),

    fillCHShistos =cms.bool(False),
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
        bypassAllPVChecks = cms.bool(True),
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
    #for PFJets: FIRSTDATA or RUNIISTARTUP (suitable for RECO beyond 7_2_X) or WINTER16 (for 8_0_X onwards)
    JetIDVersion               = cms.string("WINTER16"),
    JetType = cms.string('pf'),#pf, calo or jpt
    JetCorrections = cms.InputTag("dqmAk4PFL1FastL2L3ResidualCorrector"),
    jetsrc = cms.InputTag("ak4PFJets"),
    METCollectionLabel     = cms.InputTag("pfMet"),
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
        ),
    METCollectionLabel     = cms.InputTag("pfMet"),
)

jetDQMAnalyzerAk4PFCHSCleaned=jetDQMAnalyzerAk4PFCleaned.clone(
    filljetHighLevel =cms.bool(True),
    JetCorrections = cms.InputTag("dqmAk4PFCHSL1FastL2L3ResidualCorrector"),
    jetsrc = cms.InputTag("ak4PFJetsCHS"),
    METCollectionLabel     = cms.InputTag("pfMETT1"),
    InputMVAPUIDDiscriminant = cms.InputTag("pileupJetIdEvaluatorCHSDQM","fullDiscriminant"),
    InputCutPUIDDiscriminant = cms.InputTag("pileupJetIdEvaluatorCHSDQM","cutbasedDiscriminant"),
    InputMVAPUIDValue = cms.InputTag("pileupJetIdEvaluatorCHSDQM","fullId"),
    InputCutPUIDValue = cms.InputTag("pileupJetIdEvaluatorCHSDQM","cutbasedId"),
    fillCHShistos =cms.bool(True),
)

jetDQMAnalyzerAk4PFCHSUncleanedMiniAOD=jetDQMAnalyzerAk4PFUncleaned.clone(
    filljetHighLevel =cms.bool(True),
    CleaningParameters = cleaningParameters.clone(
        vertexCollection    = cms.InputTag( "goodOfflinePrimaryVerticesDQMforMiniAOD" ),
        ),
    JetType = cms.string('miniaod'),#pf, calo or jpt
    jetsrc = cms.InputTag("slimmedJets"),
    METCollectionLabel     = cms.InputTag("slimmedMETs"),
)

jetDQMAnalyzerAk4PFCHSCleanedMiniAOD=jetDQMAnalyzerAk4PFCleaned.clone(
    CleaningParameters = cleaningParameters.clone(
        vertexCollection    = cms.InputTag( "goodOfflinePrimaryVerticesDQMforMiniAOD" ),
        ),
    JetType = cms.string('miniaod'),#pf, calo or jpt
    jetsrc = cms.InputTag("slimmedJets"),
)

jetDQMAnalyzerAk8PFPUPPICleanedMiniAOD=jetDQMAnalyzerAk4PFCHSCleanedMiniAOD.clone(
    jetsrc = cms.InputTag("slimmedJetsAK8"),
    fillsubstructure =cms.bool(True),
)

jetDQMAnalyzerAk4PFCHSPuppiCleanedMiniAOD=jetDQMAnalyzerAk4PFCHSCleanedMiniAOD.clone(
    JetType = cms.string('miniaod'),#pf, calo or jpt
    jetsrc = cms.InputTag("slimmedJetsPuppi"),
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
                                       centralitycollection = cms.InputTag("hiCentrality"),
                                       centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                         centralitycollection = cms.InputTag("hiCentrality"),
                                         centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                         centralitycollection = cms.InputTag("hiCentrality"),
                                         centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                         centralitycollection = cms.InputTag("hiCentrality"),
                                         centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                       centralitycollection = cms.InputTag("hiCentrality"),
                                       centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                       centralitycollection = cms.InputTag("hiCentrality"),
                                       centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                       centralitycollection = cms.InputTag("hiCentrality"),
                                       centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                         centralitycollection = cms.InputTag("hiCentrality"),
                                         centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                         centralitycollection = cms.InputTag("hiCentrality"),
                                         centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                         centralitycollection = cms.InputTag("hiCentrality"),
                                         centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                         centralitycollection = cms.InputTag("hiCentrality"),
                                         centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
                                         JetCorrections = cms.string(""),
                                         recoJetPtThreshold = cms.double(10),        
                                         RThreshold = cms.double(0.3),
                                         reverseEnergyFractionThreshold = cms.double(0.5)
)

jetDQMAnalyzerAkVs3PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons",
                                       JetType = cms.untracked.string('pf'),
                                       UEAlgo = cms.untracked.string('Vs'),
                                       OutputFile = cms.untracked.string(''),
                                       src = cms.InputTag("akVs3PFJets"),
                                       PFcands = cms.InputTag("particleFlowTmp"),
                                       Background = cms.InputTag("voronoiBackgroundPF"),
                                       #srcRho = cms.InputTag("akVs3PFJets","rho"),
                                       centralitycollection = cms.InputTag("hiCentrality"),
                                       centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                       centralitycollection = cms.InputTag("hiCentrality"),
                                       centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
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
                                       centralitycollection = cms.InputTag("hiCentrality"),
                                       centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
                                       JetCorrections = cms.string(""),
                                       recoJetPtThreshold = cms.double(10),        
                                       RThreshold = cms.double(0.3),
                                       reverseEnergyFractionThreshold = cms.double(0.5)
)


jetDQMMatchAkPu3CaloAkVs3Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                               src_Jet1 = cms.InputTag("akPu3CaloJets"),
                                               src_Jet2 = cms.InputTag("akVs3CaloJets"),
                                               Jet1     = cms.untracked.string("PuCalo"),
                                               Jet2     = cms.untracked.string("VsCalo"),
                                               recoJetPtThreshold = cms.double(20.),
                                               recoDelRMatch = cms.double(0.2),
                                               recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkPu3PFAkVs3PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                           src_Jet1 = cms.InputTag("akPu3PFJets"),
                                           src_Jet2 = cms.InputTag("akVs3PFJets"),
                                           Jet1     = cms.untracked.string("PuPF"),
                                           Jet2     = cms.untracked.string("VsPF"),
                                           recoJetPtThreshold = cms.double(20.),
                                           recoDelRMatch = cms.double(0.2),
                                           recoJetEtaCut = cms.double(2.0)
)

jetDQMMatchAkPu3CaloAkPu3PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                             src_Jet1 = cms.InputTag("akPu3CaloJets"),
                                             src_Jet2 = cms.InputTag("akPu3PFJets"),
                                             Jet1     = cms.untracked.string("PuCalo"),
                                             Jet2     = cms.untracked.string("PuPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkVs3CaloAkVs3PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                             src_Jet1 = cms.InputTag("akVs3CaloJets"),
                                             src_Jet2 = cms.InputTag("akVs3PFJets"),
                                             Jet1     = cms.untracked.string("VsCalo"),
                                             Jet2     = cms.untracked.string("VsPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)

jetDQMMatchAkPu4CaloAkVs4Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                               src_Jet1 = cms.InputTag("akPu4CaloJets"),
                                               src_Jet2 = cms.InputTag("akVs4CaloJets"),
                                               Jet1     = cms.untracked.string("PuCalo"),
                                               Jet2     = cms.untracked.string("VsCalo"),
                                               recoJetPtThreshold = cms.double(20.),
                                               recoDelRMatch = cms.double(0.2),
                                               recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkPu4PFAkVs4PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                           src_Jet1 = cms.InputTag("akPu4PFJets"),
                                           src_Jet2 = cms.InputTag("akVs4PFJets"),
                                           Jet1     = cms.untracked.string("PuPF"),
                                           Jet2     = cms.untracked.string("VsPF"),
                                           recoJetPtThreshold = cms.double(20.),
                                           recoDelRMatch = cms.double(0.2),
                                           recoJetEtaCut = cms.double(2.0)
)

jetDQMMatchAkPu4CaloAkPu4PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                             src_Jet1 = cms.InputTag("akPu4CaloJets"),
                                             src_Jet2 = cms.InputTag("akPu4PFJets"),
                                             Jet1     = cms.untracked.string("PuCalo"),
                                             Jet2     = cms.untracked.string("PuPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkVs4CaloAkVs4PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                             src_Jet1 = cms.InputTag("akVs4CaloJets"),
                                             src_Jet2 = cms.InputTag("akVs4PFJets"),
                                             Jet1     = cms.untracked.string("VsCalo"),
                                             Jet2     = cms.untracked.string("VsPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkPu5CaloAkVs5Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                               src_Jet1 = cms.InputTag("akPu5CaloJets"),
                                               src_Jet2 = cms.InputTag("akVs5CaloJets"),
                                               Jet1     = cms.untracked.string("PuCalo"),
                                               Jet2     = cms.untracked.string("VsCalo"),
                                               recoJetPtThreshold = cms.double(20.),
                                               recoDelRMatch = cms.double(0.2),
                                               recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkPu5PFAkVs5PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                           src_Jet1 = cms.InputTag("akPu5PFJets"),
                                           src_Jet2 = cms.InputTag("akVs5PFJets"),
                                           Jet1     = cms.untracked.string("PuPF"),
                                           Jet2     = cms.untracked.string("VsPF"),
                                           recoJetPtThreshold = cms.double(20.),
                                           recoDelRMatch = cms.double(0.2),
                                           recoJetEtaCut = cms.double(2.0)
)

jetDQMMatchAkPu5CaloAkPu5PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                             src_Jet1 = cms.InputTag("akPu5CaloJets"),
                                             src_Jet2 = cms.InputTag("akPu5PFJets"),
                                             Jet1     = cms.untracked.string("PuCalo"),
                                             Jet2     = cms.untracked.string("PuPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkVs5CaloAkVs5PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                             src_Jet1 = cms.InputTag("akVs5CaloJets"),
                                             src_Jet2 = cms.InputTag("akVs5PFJets"),
                                             Jet1     = cms.untracked.string("VsCalo"),
                                             Jet2     = cms.untracked.string("VsPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkPu2CaloAkVs2Calo = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                               src_Jet1 = cms.InputTag("akPu2CaloJets"),
                                               src_Jet2 = cms.InputTag("akVs2CaloJets"),
                                               Jet1     = cms.untracked.string("PuCalo"),
                                               Jet2     = cms.untracked.string("VsCalo"),
                                               recoJetPtThreshold = cms.double(20.),
                                               recoDelRMatch = cms.double(0.2),
                                               recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkPu2PFAkVs2PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                           src_Jet1 = cms.InputTag("akPu2PFJets"),
                                           src_Jet2 = cms.InputTag("akVs2PFJets"),
                                           Jet1     = cms.untracked.string("PuPF"),
                                           Jet2     = cms.untracked.string("VsPF"),
                                           recoJetPtThreshold = cms.double(20.),
                                           recoDelRMatch = cms.double(0.2),
                                           recoJetEtaCut = cms.double(2.0)
)

jetDQMMatchAkPu2CaloAkPu2PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                             src_Jet1 = cms.InputTag("akPu2CaloJets"),
                                             src_Jet2 = cms.InputTag("akPu2PFJets"),
                                             Jet1     = cms.untracked.string("PuCalo"),
                                             Jet2     = cms.untracked.string("PuPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)
jetDQMMatchAkVs2CaloAkVs2PF = cms.EDAnalyzer("JetAnalyzer_HeavyIons_matching",
                                             src_Jet1 = cms.InputTag("akVs2CaloJets"),
                                             src_Jet2 = cms.InputTag("akVs2PFJets"),
                                             Jet1     = cms.untracked.string("VsCalo"),
                                             Jet2     = cms.untracked.string("VsPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)
