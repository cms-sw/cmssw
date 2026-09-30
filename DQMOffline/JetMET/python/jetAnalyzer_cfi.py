import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetDQMConfig_cff import *      # parameters for all jet analyzers
from DQMOffline.JetMET.jetMETDQMCleanup_cff import *  # parameters for event cleanup

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
jetDQMAnalyzerAk4CaloUncleaned = DQMEDAnalyzer('JetAnalyzer',
    disabledMEs = cms.vstring(),
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
        andOr          = cms.bool( False ),
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltPaths       = cms.vstring( 'HLT_PFJet450_v*'), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    lowPtJetTrigger = cms.PSet(
        andOr          = cms.bool( False ),
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
        bypassAllPVChecks = True,
        ),

    #Only for JPT and CaloJetID  -> need to define InputJetIDValueMap  
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
    ),
    stage2L1 = cms.bool(False) # by default use legacy L1T code
)

## change the default to stage2 for all current eras
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(jetDQMAnalyzerAk4CaloUncleaned,
                         stage2L1 = True,
                         highPtJetTrigger = dict(stage2 = cms.bool(True),
                                                 l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                 l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                 ReadPrescalesFromFile = cms.bool(False)),
                         lowPtJetTrigger = dict(stage2 = cms.bool(True),
                                                l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                ReadPrescalesFromFile = cms.bool(False)))

jetDQMAnalyzerAk4CaloCleaned=jetDQMAnalyzerAk4CaloUncleaned.clone(
    JetCleaningFlag   = True,
    filljetHighLevel  = False,
    CleaningParameters = cleaningParameters.clone(
        bypassAllPVChecks = True,
    ),
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = 20.,
        asymmetryThirdJetCut = 30,
        balanceThirdJetCut   = 0.2, 
       )  
)

jetDQMAnalyzerAk4PFUncleaned=jetDQMAnalyzerAk4CaloUncleaned.clone(
    CleaningParameters = cleaningParameters.clone(
       bypassAllPVChecks  = False,
    ),
    #for PFJets: LOOSE,TIGHT
    JetIDQuality               = cms.string("TIGHT"),
    #options for Calo and JPT: PURE09,DQM09,CRAFT08
    #for PFJets: RUN2ULCHS for 11_1_X onwards
    JetIDVersion               = cms.string("RUN2ULCHS"),
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
    JetCleaningFlag = True,
    filljetHighLevel = False,
    jetAnalysis=jetDQMParameters.clone(
        ptThreshold = 20.,
        asymmetryThirdJetCut = 30,
        balanceThirdJetCut = 0.2,
        ),
    METCollectionLabel     = "pfMet"
)

jetDQMAnalyzerAk4PFCHSCleaned=jetDQMAnalyzerAk4PFCleaned.clone(
    filljetHighLevel = True,
    JetCorrections = "dqmAk4PFCHSL1FastL2L3ResidualCorrector",
    jetsrc = "ak4PFJetsCHS",
    METCollectionLabel     = "pfMETT1",
    InputMVAPUIDDiscriminant = "pileupJetIdEvaluatorCHSDQM:fullDiscriminant",
    InputCutPUIDDiscriminant = "pileupJetIdEvaluatorCHSDQM:cutbasedDiscriminant",
    InputMVAPUIDValue = "pileupJetIdEvaluatorCHSDQM:fullId",
    InputCutPUIDValue = "pileupJetIdEvaluatorCHSDQM:cutbasedId",
    fillCHShistos = True
)

jetDQMAnalizerAk4PUPPICleaned=jetDQMAnalyzerAk4PFCleaned.clone(
    JetType = cms.string('puppi'),
    jetsrc = "ak4PFJetsPuppi",
    METCollectionLabel = "pfMetPuppi",
    JetCorrections = cms.InputTag("dqmAk4PFPuppiL1FastL2L3ResidualCorrector"),
    JetIDVersion = "RUN2ULPUPPI",
    JetIDQuality = cms.string("TIGHT"),
    fillCHShistos = True,
)

jetDQMAnalyzerAk4ScoutingUncleaned = jetDQMAnalyzerAk4CaloUncleaned.clone(  
    JetType = cms.string('scouting'),
    JetCorrections = cms.InputTag("dqmAk4PFScoutingL1FastL2L3ResidualCorrector"),
    jetsrc = cms.InputTag("hltScoutingPFPacker"),                       ###---> this name goes to GUI and TTree under .../JetMET/Run summary/Jet
    srcRho = cms.InputTag("hltScoutingPFPacker","rho"),
    METCollectionLabel = cms.InputTag("hltScoutingPFPacker","pfMetPt"), ###---> this name goes to GUI and TTree under .../JetMET/Run summary/MET 
    muonsrc = cms.InputTag("hltScoutingMuonPackerNoVtx","","HLT"),
    l1algoname = cms.string("L1Tech_BPTX_plus_AND_minus.v0"),
    filljetHighLevel = False,                                           ### for plots: "vertices" and "cleanup" in .../JetMET/Run summary/
    fillsubstructure = False,
    
    highPtJetTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltPaths       = cms.vstring( 'DST_PFScouting_JetHT_v*'), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    lowPtJetTrigger = cms.PSet(
        andOr         = cms.bool( False ),
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltPaths       = cms.vstring( 'DST_PFScouting_ZeroBias_v*'), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),

    TriggerResultsLabel        = cms.InputTag("TriggerResults::HLT"),
    processname                = cms.string("HLT"),                     

    #
    # Jet-related
    #   

    JetCleaningFlag            = cms.untracked.bool(False),       

    runcosmics                 = False,                
                                
    #Cleanup parameters
    CleaningParameters = cleaningParameters.clone(
        bypassAllPVChecks = False,
        ),

    #for scouting PF jets: TIGHT
    JetIDQuality               = cms.string("TIGHT"),
    #for scouting PF jets: RUN3Scouting
    JetIDVersion               = cms.string("RUN3Scouting"),
    #
    #Pileup JetID anf quark-gluon discrimination not exist for scouting PF jets. The following: actually done only for PFJets at the moment
    ###InputMVAPUIDDiscriminant = cms.InputTag("pileupJetIdEvaluatorDQM","fullDiscriminant"),
    ###InputCutPUIDDiscriminant = cms.InputTag("pileupJetIdEvaluatorDQM","cutbasedDiscriminant"),
    ###InputMVAPUIDValue = cms.InputTag("pileupJetIdEvaluatorDQM","fullId"),
    ###InputCutPUIDValue = cms.InputTag("pileupJetIdEvaluatorDQM","cutbasedId"),

    ###InputQGMultiplicity = cms.InputTag("QGTagger", "mult"),
    ###InputQGLikelihood = cms.InputTag("QGTagger", "qgLikelihood"),
    ###InputQGPtDToken = cms.InputTag("QGTagger", "ptD"),
    ###InputQGAxis2 = cms.InputTag("QGTagger", "axis2"),

    fillCHShistos = False,
    #
    # For jetAnalysis
    #
    jetAnalysis = jetDQMParameters.clone(),

    #
    # DCS ### -> only used in JetMETDQMFilter.cc
    #                             
    DCSFilterForJetMonitoring = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
      #DebugOn = cms.untracked.bool(True),
      alwaysPass = cms.untracked.bool(False)
    )
)

stage2L1Trigger.toModify(jetDQMAnalyzerAk4ScoutingUncleaned,
                         stage2L1 = True,
                         highPtJetTrigger = dict(stage2 = cms.bool(True),
                                                 l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                 l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                 ReadPrescalesFromFile = cms.bool(False)),
                         lowPtJetTrigger = dict(stage2 = cms.bool(True),
                                                l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                ReadPrescalesFromFile = cms.bool(False)))

jetDQMAnalyzerAk4ScoutingCleaned = jetDQMAnalyzerAk4ScoutingUncleaned.clone(
    JetCleaningFlag = True
)

jetDQMAnalyzerAk4PFCHSUncleanedMiniAOD=jetDQMAnalyzerAk4PFUncleaned.clone(
    filljetHighLevel = True,
    CleaningParameters = cleaningParameters.clone(
        vertexCollection  =  "goodOfflinePrimaryVerticesDQMforMiniAOD" ,
        ),
    JetType = 'miniaod',#pf, calo or jpt
    jetsrc = "slimmedJets",
    METCollectionLabel     = "slimmedMETs"
)

jetDQMAnalyzerAk4PFCHSCleanedMiniAOD=jetDQMAnalyzerAk4PFCleaned.clone(
    CleaningParameters = cleaningParameters.clone(
        vertexCollection    =  "goodOfflinePrimaryVerticesDQMforMiniAOD" 
        ),
    JetType = 'miniaod',#pf, calo or jpt
    jetsrc = "slimmedJets"
)

jetDQMAnalyzerAk8PFPUPPICleanedMiniAOD=jetDQMAnalyzerAk4PFCHSCleanedMiniAOD.clone(
    jetsrc = "slimmedJetsAK8",
    #for PUPPI jets: TIGHT
    JetIDQuality  = "TIGHT",
    #for PUPPI jets: RUN2ULPUPPI from 11_1_X onwards
    JetIDVersion  = "RUN2ULPUPPI",
    fillsubstructure =True
)

jetDQMAnalyzerAk4PFCHSPuppiCleanedMiniAOD=jetDQMAnalyzerAk4PFCHSCleanedMiniAOD.clone(
    JetType = 'miniaod',#pf, calo or jpt
    jetsrc = "slimmedJetsPuppi",
    #for PUPPI jets: TIGHT
    JetIDQuality  = "TIGHT",
    #for PUPPI jets: RUN2ULPUPPI from 11_1_X onwards
    JetIDVersion  = "RUN2ULPUPPI"
)

jetDQMAnalyzerIC5CaloHIUncleaned=jetDQMAnalyzerAk4CaloUncleaned.clone(
    filljetHighLevel = True,
    CleaningParameters = cleaningParameters.clone(
        bypassAllPVChecks  = False,
        vertexCollection =  "hiSelectedVertex",
        ),
    JetType = 'calo',#pf, calo or jpt
    JetCorrections = "",# no jet correction available yet?
    jetsrc = "iterativeConePu5CaloJets",
    JetCleaningFlag            = False,  
    runcosmics                 = True,   
    DCSFilterForJetMonitoring = dict(
        DetectorTypes = "ecal:hbhe:hf:pixel:sistrip:es:muon",
        #DebugOn = True,
        alwaysPass = False
    )
)

jetDQMAnalyzerAkPU3Calo = DQMEDAnalyzer('JetAnalyzer_HeavyIons',
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
jetDQMAnalyzerAkPU4Calo = jetDQMAnalyzerAkPU3Calo.clone(src = "akPu4CaloJets")
jetDQMAnalyzerAkPU5Calo = jetDQMAnalyzerAkPU3Calo.clone(src = "akPu5CaloJets")
 
jetDQMAnalyzerAkPU3PF = DQMEDAnalyzer('JetAnalyzer_HeavyIons',
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
jetDQMAnalyzerAkPU4PF = jetDQMAnalyzerAkPU3PF.clone(src = "akPu4PFJets")
jetDQMAnalyzerAkPU5PF = jetDQMAnalyzerAkPU3PF.clone(src = "akPu5PFJets")

jetDQMAnalyzerAkCs3PF = DQMEDAnalyzer('JetAnalyzer_HeavyIons',
                                         JetType = cms.untracked.string('pf'),
                                         UEAlgo = cms.untracked.string('Cs'),
                                         OutputFile = cms.untracked.string(''),
                                         src = cms.InputTag("akCs3PFJets"),
                                         CScands = cms.InputTag("akCs3PFJets","pfParticlesCs"),
                                         PFcands = cms.InputTag("particleFlowTmp"),
                                         centralitycollection = cms.InputTag("hiCentrality"),
                                         #centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
                                         JetCorrections = cms.string(""),
                                         recoJetPtThreshold = cms.double(10),        
                                         RThreshold = cms.double(0.3),
                                         reverseEnergyFractionThreshold = cms.double(0.5),
                                         etaMap    = cms.InputTag('hiFJRhoProducer','mapEtaEdges'),
                                         rho       = cms.InputTag('hiFJRhoProducer','mapToRho'),
                                         rhom      = cms.InputTag('hiFJRhoProducer','mapToRhoM')
)

jetDQMAnalyzerAkCs4PF=jetDQMAnalyzerAkCs3PF.clone(src = "akCs4PFJets",
						  CScands = "akCs4PFJets:pfParticlesCs"
)


jetDQMMatchAkPu3CaloAkPu3PF = DQMEDAnalyzer('JetAnalyzer_HeavyIons_matching',
                                             src_Jet1 = cms.InputTag("akPu3CaloJets"),
                                             src_Jet2 = cms.InputTag("akPu3PFJets"),
                                             Jet1     = cms.untracked.string("PuCalo"),
                                             Jet2     = cms.untracked.string("PuPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)

jetDQMMatchAkPu4CaloAkPu4PF = DQMEDAnalyzer('JetAnalyzer_HeavyIons_matching',
                                             src_Jet1 = cms.InputTag("akPu4CaloJets"),
                                             src_Jet2 = cms.InputTag("akPu4PFJets"),
                                             Jet1     = cms.untracked.string("PuCalo"),
                                             Jet2     = cms.untracked.string("PuPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)

jetDQMMatchAkPu5CaloAkPu5PF = DQMEDAnalyzer('JetAnalyzer_HeavyIons_matching',
                                             src_Jet1 = cms.InputTag("akPu5CaloJets"),
                                             src_Jet2 = cms.InputTag("akPu5PFJets"),
                                             Jet1     = cms.untracked.string("PuCalo"),
                                             Jet2     = cms.untracked.string("PuPF"),
                                             recoJetPtThreshold = cms.double(20.),
                                             recoDelRMatch = cms.double(0.2),
                                             recoJetEtaCut = cms.double(2.0)
)

# Exact base-folder selections from the Phase2_cleanup/DC inventory (2026-09-30).
# Set the final instances after all clones so CHS, MiniAOD, scouting and cosmics
# retain their own monitoring. HGCAL retirements are restricted to phase2_hgcal.
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal

_caloJetDisabledMEs = (
    'Eta_Barrel',
    'Eta_Barrel_Hi',
    'Eta_Forward',
    'Eta_Forward_Hi',
    'Eta_Hi',
    'JetArea_uncor',
    'NJets_Hi',
    'Phi_Barrel_Hi',
    'Phi_Forward_Hi',
    'Phi_Hi',
    'Phi_Lo',
    'Pt_Barrel_Hi',
    'Pt_Forward_Hi',
    'Pt_Hi',
    'Pt_Lo',
)

_caloJetHGCALDisabledMEs = (
    'Constituents_EndCap',
    'EFrac_EndCap',
    'Eta_EndCap',
    'Eta_EndCap_Hi',
    'HFrac_EndCap',
    'Phi_EndCap',
    'Phi_EndCap_Hi',
    'Pt_EndCap',
    'Pt_EndCap_Hi',
)

jetDQMAnalyzerAk4CaloCleaned.disabledMEs = cms.vstring(*_caloJetDisabledMEs)
phase2_hgcal.toModify(
    jetDQMAnalyzerAk4CaloCleaned,
    disabledMEs = cms.vstring(*(_caloJetDisabledMEs + _caloJetHGCALDisabledMEs)),
)

_pfJetDisabledMEs = (
    'CHFracBarrel_BXm1Empty',
    'CHFracBarrel_BXm1Filled',
)

_pfJetHGCALDisabledMEs = (
    'CHEn_highPt_EndCap',
    'CHEn_lowPt_EndCap',
    'CHEn_mediumPt_EndCap',
    'CHFracEndCapMinus_BXm1Empty',
    'CHFracEndCapMinus_BXm1Filled',
    'CHFracEndCapPlus_BXm1Empty',
    'CHFracEndCapPlus_BXm1Filled',
    'CHFracVSpT_EndCap',
    'CHFrac_highPt_EndCap',
    'CHFrac_lowPt_EndCap',
    'CHFrac_mediumPt_EndCap',
    'ChMultiplicity_highPt_EndCap',
    'ChMultiplicity_lowPt_EndCap',
    'ChMultiplicity_mediumPt_EndCap',
    'Constituents_EndCap',
    'CutPUJIDDiscriminant_highPt_EndCap',
    'CutPUJIDDiscriminant_lowPt_EndCap',
    'CutPUJIDDiscriminant_mediumPt_EndCap',
    'ElEn_highPt_EndCap',
    'ElEn_lowPt_EndCap',
    'ElEn_mediumPt_EndCap',
    'Eta_EndCap',
    'Eta_EndCap_Hi',
    'JetMass_highPt_EndCap',
    'JetMass_lowPt_EndCap',
    'JetMass_mediumPt_EndCap',
    'MVAPUJIDDiscriminant_highPt_EndCap',
    'MVAPUJIDDiscriminant_lowPt_EndCap',
    'MVAPUJIDDiscriminant_mediumPt_EndCap',
    'MuEn_highPt_EndCap',
    'MuEn_lowPt_EndCap',
    'MuEn_mediumPt_EndCap',
    'MuMultiplicity_highPt_EndCap',
    'MuMultiplicity_lowPt_EndCap',
    'MuMultiplicity_mediumPt_EndCap',
    'NHEn_highPt_EndCap',
    'NHEn_lowPt_EndCap',
    'NHEn_mediumPt_EndCap',
    'NHFracEndCapMinus_BXm1Empty',
    'NHFracEndCapMinus_BXm1Filled',
    'NHFracEndCapPlus_BXm1Empty',
    'NHFracEndCapPlus_BXm1Filled',
    'NHFracVSpT_EndCap',
    'NHFrac_highPt_EndCap',
)

jetDQMAnalyzerAk4PFCleaned.disabledMEs = cms.vstring(*_pfJetDisabledMEs)
phase2_hgcal.toModify(
    jetDQMAnalyzerAk4PFCleaned,
    disabledMEs = cms.vstring(*(_pfJetDisabledMEs + _pfJetHGCALDisabledMEs)),
)

_puppiJetDisabledMEs = (
    'CHFracVSpT_Barrel',
    'CHFracVSpT_EndCap',
    'HFEFracVSpT_Forward',
    'HFHFracVSpT_Forward',
    'JetIDPassFractionVSpt',
    'JetIDPassFractionVSeta',
    'JetIDPassFractionVSptNoHF',
    'NHFracVSpT_Barrel',
    'NHFracVSpT_EndCap',
    'PhFracVSpT_Barrel',
    'PhFracVSpT_EndCap',
    'Pt_Barrel_Hi',
    'Pt_Forward_Hi',
    'Pt_Hi',
    'Pt_Lo',
    'MVAPUJIDDiscriminant_highPt_Barrel',
    'MVAPUJIDDiscriminant_highPt_EndCap',
    'MVAPUJIDDiscriminant_highPt_Forward',
    'MVAPUJIDDiscriminant_lowPt_Barrel',
    'MVAPUJIDDiscriminant_lowPt_EndCap',
    'MVAPUJIDDiscriminant_lowPt_Forward',
    'MVAPUJIDDiscriminant_mediumPt_Barrel',
    'MVAPUJIDDiscriminant_mediumPt_EndCap',
    'MVAPUJIDDiscriminant_mediumPt_Forward',
    'NJets_Hi',
    'Phi_Barrel_Hi',
    'Phi_EndCap_Hi',
    'Phi_Forward_Hi',
    'Phi_Hi',
    'Phi_Lo',
)

# The five PUPPI names below are conditional Discard/replace for HGCAL decisions.
_puppiJetHGCALDisabledMEs = (
    'Eta_EndCap',
    'Eta_EndCap_Hi',
    'Pt_EndCap',
    'Pt_EndCap_Hi',
    'Constituents_EndCap',
)

jetDQMAnalizerAk4PUPPICleaned.disabledMEs = cms.vstring(*_puppiJetDisabledMEs)
phase2_hgcal.toModify(
    jetDQMAnalizerAk4PUPPICleaned,
    disabledMEs = cms.vstring(*(_puppiJetDisabledMEs + _puppiJetHGCALDisabledMEs)),
)
