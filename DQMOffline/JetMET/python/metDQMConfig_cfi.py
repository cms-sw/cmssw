import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETDQMCleanup_cff import *

tcMetAnalyzer = cms.EDAnalyzer("METAnalyzer",

    OutputFile = cms.string('jetMETMonitoring.root'),
    METType=cms.untracked.string('tc'),

    METCollectionLabel     = cms.InputTag("tcMet"),
    JetCollectionLabel  = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    
    FolderName = cms.untracked.string("JetMET/MET/"),

    CleaningParameters = cleaningParameters.clone(),

    TriggerResultsLabel  = cms.InputTag("TriggerResults::HLT"),

    HLTPathsJetMB = cms.vstring(),
#    When it is empty, it accepts all the triggers

    triggerSelectedSubFolders = cms.VPSet(
    cms.PSet( label = cms.string('highPtJet'),
        andOr         = cms.bool( False ),    #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_highptjet' ), #overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_PFJet400' ), 
        andOrHlt       = cms.bool( True ),#ineffective: Always OR
        errorReplyHlt  = cms.bool( False ),
    ),
    cms.PSet(label = cms.string('lowPtJet'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_lowptjet' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_PFJet80' ), 
        andOrHlt       = cms.bool( True ),#ineffective: Always OR
        errorReplyHlt  = cms.bool( True ),
    ),
    cms.PSet(label = cms.string('zeroBias'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_minbias' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_ZeroBias' ), 
        andOrHlt       = cms.bool( True ),#ineffective: Always OR
        errorReplyHlt  = cms.bool( False ),
    ),
    cms.PSet(label = cms.string('highMET'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_highmet' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_MET400' ), 
        andOrHlt       = cms.bool( True ),#ineffective: Always OR
        errorReplyHlt  = cms.bool( False ),
    ),
    cms.PSet(label = cms.string('singleEle'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_ele' ),#overrides hltPaths!
        hltPaths       = cms.vstring('HLT_Ele27_WP80' ), #ineffective: Always OR
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
    ),
    cms.PSet(label = cms.string('singleMu'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_muon' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_IsoMu24_eta2p1', 'HLT_IsoMu24'), 
        andOrHlt       = cms.bool( True ),#ineffective: Always OR
        errorReplyHlt  = cms.bool( False ),
    ) 
    ),

    JetIDParams = cms.PSet(
        useRecHits = cms.bool(True),
        hbheRecHitsColl = cms.InputTag("hbhereco"),
        hoRecHitsColl   = cms.InputTag("horeco"),
        hfRecHitsColl   = cms.InputTag("hfreco"),
        ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
        eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
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

pfMetAnalyzer = tcMetAnalyzer.clone(
    METType=cms.untracked.string('pf'),
    METCollectionLabel     = cms.InputTag("pfMet"),
    JetCollectionLabel  = cms.InputTag("ak5PFJets"),
)

metAnalyzer = tcMetAnalyzer.clone(
    METType=cms.untracked.string('calo'),
    METCollectionLabel     = cms.InputTag("met"),
    JetCollectionLabel  = cms.InputTag("ak5CaloJets"),
    DCSFilter = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    )
)
