import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETDQMCleanup_cff import *

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5CaloL2L3,ak5CaloL2Relative,ak5CaloL3Absolute
newAk5CaloL2L3 = ak5CaloL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak7CaloL2L3,ak7CaloL2Relative,ak7CaloL3Absolute
newAk7CaloL2L3 = ak7CaloL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5PFL1FastL2L3,ak5PFL1Fastjet,ak5PFL2Relative,ak5PFL3Absolute
newAk5PFL1FastL2L3 = ak5PFL1FastL2L3.clone()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5JPTL1FastL2L3,ak5JPTL1Fastjet,ak5JPTL2Relative,ak5JPTL3Absolute
newAk5JPTL1FastL2L3 = ak5JPTL1FastL2L3.clone()


tcMetDQMAnalyzer = cms.EDAnalyzer("METAnalyzer",
    OutputMEsInRootFile = cms.bool(False),
    OutputFile = cms.string('jetMETMonitoring.root'),
    METType=cms.untracked.string('tc'),
    METCollectionLabel     = cms.InputTag("tcMet"),

    JetCollectionLabel  = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    JetCorrections = cms.string("newAk5JPTL1FastL2L3"),
    InputJetIDValueMap         = cms.InputTag("ak5JetID"), 
    ptThreshold                =cms.double(30),
    
    FolderName = cms.untracked.string("JetMET/MET/"),

    CleaningParameters = cleaningParameters.clone(),

    TriggerResultsLabel  = cms.InputTag("TriggerResults::HLT"),

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
        hltDBKey       = cms.string( 'jetmet_minbias' ),#overrides hltPaths!
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
    cms.PSet(label = cms.string('singleEle'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_ele' ),#overrides hltPaths!
        hltPaths       = cms.vstring('HLT_Ele27_WP80_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
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
    HBHENoiseFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHENoiseFilterResult"),
    BeamHaloSummaryLabel       = cms.InputTag("BeamHaloSummary"),    

#    HighPtJetThreshold = cms.double(60.),
#    LowPtJetThreshold  = cms.double(15.),
#    HighMETThreshold   = cms.double(110.),

    pVBin       = cms.int32(100),
    pVMax       = cms.double(100.0),
    pVMin       = cms.double(0.0),

    verbose     = cms.int32(0),

#    etThreshold  = cms.double(2.),

    DCSFilter = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        #DebugOn = cms.untracked.bool(True),
        Filter = cms.untracked.bool(True)
        ),
    
    #Parameters set for METAnalyzer --> but only used for TCMET
    InputBeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    InputTrackLabel    = cms.InputTag("generalTracks"),
    InputMuonLabel     = cms.InputTag("muons"),
    InputElectronLabel = cms.InputTag("gedGsfElectrons"),
    InputTCMETValueMap = cms.InputTag("muonTCMETValueMapProducer","muCorrData"),#muonMETValueMapProducer -> calomet vs muonTCMETValueMapProducer
)

pfMetDQMAnalyzer = tcMetDQMAnalyzer.clone(
    METType=cms.untracked.string('pf'),
    METCollectionLabel     = cms.InputTag("pfMet"),
    JetCollectionLabel  = cms.InputTag("ak5PFJets"),
    JetCorrections = cms.string("newAk5PFL1FastL2L3"),
)

caloMetDQMAnalyzer = tcMetDQMAnalyzer.clone(
    METType=cms.untracked.string('calo'),
    METCollectionLabel     = cms.InputTag("met"),
    JetCollectionLabel  = cms.InputTag("ak5CaloJets"),
    JetCorrections = cms.string("newAk5CaloL2L3"),
    DCSFilter = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    )
)
