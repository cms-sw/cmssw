import FWCore.ParameterSet.Config as cms

SingleTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/SingleTau/Path'),
    L1SeedFilter          = cms.untracked.InputTag("hltL1sSingleTau","","HLT"),
    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
    MuonFilter            = cms.untracked.InputTag("DUMMY"),
    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
    NTriggeredTaus        = cms.untracked.uint32(1),
    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(''),
    LogFileName           = cms.untracked.string(''),
    MatchDeltaRL1         = cms.untracked.double(0.5),
    MatchDeltaRHLT        = cms.untracked.double(0.3)
)


SingleTauL2Val = cms.EDFilter("HLTTauCaloDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/SingleTau/L2'),
    L2InfoAssociationInput = cms.InputTag("hltL2SingleTauIsolationProducer","L2TauIsolationInfoAssociator"),
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    MET                    = cms.InputTag("hltMet"),
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    L2IsolatedJets         = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(200.),
    NBins                  = cms.int32(50)                            
)


SingleTauL25Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/SingleTau/L25'),
    ConeIsolation          = cms.InputTag("hltConeIsolationL25SingleTau"),
    InputJets              = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),                             
    IsolatedJets           = cms.InputTag("hltIsolatedL25SingleTau"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L25'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(200.),
    NBins                  = cms.int32(50)                            
)


SingleTauL3Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/SingleTau/L3'),
    ConeIsolation          = cms.InputTag("hltConeIsolationL3SingleTau"),
    InputJets              = cms.InputTag("hltIsolatedL25SingleTau"),                             
    IsolatedJets           = cms.InputTag("hltIsolatedL3SingleTau"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L3'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(200.),
    NBins                  = cms.int32(50)                            
)



SingleTauValidation = cms.Sequence(SingleTauPathVal + SingleTauL2Val + SingleTauL25Val+SingleTauL3Val)


