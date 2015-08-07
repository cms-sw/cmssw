import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETDQMCleanup_cff import *
from DQMOffline.JetMET.metDiagnosticParameterSet_cfi import *

#jet corrector defined in jetMETDQMOfflineSource python file

caloMetDQMAnalyzer = cms.EDAnalyzer("METAnalyzer",  
    METType=cms.untracked.string('calo'),
    srcPFlow = cms.InputTag('particleFlow', ''),
    l1algoname = cms.string("L1Tech_BPTX_plus_AND_minus.v0"),
    METCollectionLabel     = cms.InputTag("caloMet"),
    JetCollectionLabel  = cms.InputTag("ak4CaloJets"),
    JetCorrections = cms.InputTag("dqmAk4PFL1FastL2L3ResidualCorrector"),
    muonsrc = cms.InputTag("muons"),

    ptMinCand      = cms.double(1.),
    hcalMin      =cms.double(1.),

    InputJetIDValueMap         = cms.InputTag("ak4JetID"), 
    ptThreshold                =cms.double(30),
    
    FolderName = cms.untracked.string("JetMET/MET/"),

    fillMetHighLevel = cms.bool(True),

    fillCandidateMaps = cms.bool(False),

    CleaningParameters = cleaningParameters.clone(),
    METDiagonisticsParameters = multPhiCorr_METDiagnostics,

    TriggerResultsLabel  = cms.InputTag("TriggerResults::HLT"),

    onlyCleaned                = cms.untracked.bool(True),
    runcosmics                 = cms.untracked.bool(False),  

    LSBegin = cms.int32(0),
    LSEnd   = cms.int32(-1),      

#    HLTPathsJetMB = cms.vstring(),
#    When it is empty, it accepts all the triggers

    triggerSelectedSubFolders = cms.VPSet(
    cms.PSet( label = cms.string('highPtJet'),
        andOr         = cms.bool( False ),    #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_highptjet' ), #overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_PFJet400_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    cms.PSet(label = cms.string('lowPtJet'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_lowptjet' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_PFJet80_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( True ),
    ),
    cms.PSet(label = cms.string('zeroBias'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        #hltDBKey       = cms.string( 'jetmet_minbias' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_ZeroBias_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    cms.PSet(label = cms.string('highMET'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_highmet' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_MET400_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    #cms.PSet(label = cms.string('singleEle'),
    #    andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
    #    dbLabel        = cms.string("JetMETDQMTrigger"),
    #    hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_ele' ),#overrides hltPaths!
    #    hltPaths       = cms.vstring('HLT_Ele27_WP80_v*' ), 
    #    andOrHlt       = cms.bool( True ),
    #    errorReplyHlt  = cms.bool( False ),
    #),
    cms.PSet(label = cms.string('singleMu'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_muon' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_IsoMu24_eta2p1_v*', 'HLT_IsoMu24_v*'), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ) 
    ),
    
    HcalNoiseRBXCollection     = cms.InputTag("hcalnoise"),
    #HBHENoiseFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHENoiseFilterResultRun2Loose"),  
    HBHENoiseFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHENoiseFilterResult"),  

#    HighPtJetThreshold = cms.double(60.),
#    LowPtJetThreshold  = cms.double(15.),
#    HighMETThreshold   = cms.double(110.),

    #if changed here, change certification module input in same manner and injetDQMconfig
    pVBin       = cms.int32(100),
    pVMax       = cms.double(100.0),
    pVMin       = cms.double(0.0),

    verbose     = cms.int32(0),

#    etThreshold  = cms.double(2.),

    DCSFilter = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    ),
)

pfMetDQMAnalyzer = caloMetDQMAnalyzer.clone(
    METType=cms.untracked.string('pf'),
    METCollectionLabel     = cms.InputTag("pfMet"),
    srcPFlow = cms.InputTag('particleFlow', ''),
    JetCollectionLabel  = cms.InputTag("ak4PFJets"),
    JetCorrections = cms.InputTag("dqmAk4PFL1FastL2L3ResidualCorrector"),
    fillMetHighLevel = cms.bool(False),
    fillCandidateMaps = cms.bool(True),
    onlyCleaned                = cms.untracked.bool(False),
    DCSFilter = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        #DebugOn = cms.untracked.bool(True),
        Filter = cms.untracked.bool(True)
        ),
)
#both CaloMET and type1 MET only cleaned plots are filled
pfMetT1DQMAnalyzer = caloMetDQMAnalyzer.clone(
    METType=cms.untracked.string('pf'),
    METCollectionLabel     = cms.InputTag("pfMETT1"),
    srcPFlow = cms.InputTag('particleFlow', ''),
    JetCollectionLabel  = cms.InputTag("ak4PFJetsCHS"),
    JetCorrections = cms.InputTag("dqmAk4PFCHSL1FastL2L3ResidualCorrector"),
    fillMetHighLevel = cms.bool(False),
    fillCandidateMaps = cms.bool(False),
    DCSFilter = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        Filter = cms.untracked.bool(True)
        ),
)
pfMetDQMAnalyzerMiniAOD = pfMetDQMAnalyzer.clone(
    fillMetHighLevel = cms.bool(False),
    fillCandidateMaps = cms.bool(False),
    CleaningParameters = cleaningParameters.clone(
        vertexCollection    = cms.InputTag( "goodOfflinePrimaryVerticesDQMforMiniAOD" ),
        ),
    METType=cms.untracked.string('miniaod'),
    METCollectionLabel     = cms.InputTag("slimmedMETs"),
    JetCollectionLabel  = cms.InputTag("slimmedJets"),
    JetCorrections = cms.InputTag(""),#not called, since corrected by default
)
