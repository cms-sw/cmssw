import FWCore.ParameterSet.Config as cms

SingleTauMETPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/SingleTauMET/Path'),
    L1SeedFilter          = cms.untracked.InputTag("hltL1sSingleTauMET","","HLT"),
    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauMETEcalIsolation","","HLT"),
    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTauMET","","HLT"),
    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTauMET","","HLT"),
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


SingleTauMETL2Val = cms.EDFilter("HLTTauCaloDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/SingleTauMET/L2'),
    L2InfoAssociationInput = cms.InputTag("hltL2SingleTauMETIsolationProducer","L2TauIsolationInfoAssociator"),
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    MET                    = cms.InputTag("hltMet"),
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    L2IsolatedJets         = cms.InputTag("hltL2SingleTauMETIsolationSelector","Isolated"),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(200.),
    NBins                  = cms.int32(50)                            
)


SingleTauMETL25Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/SingleTauMET/L25'),
    ConeIsolation          = cms.InputTag("hltConeIsolationL25SingleTauMET"),
    InputJets              = cms.InputTag("hltL2SingleTauMETIsolationSelector","Isolated"),                             
    IsolatedJets           = cms.InputTag("hltIsolatedL25SingleTauMET"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L25'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(200.),
    NBins                  = cms.int32(50)                            
)


SingleTauMETL3Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/SingleTauMET/L3'),
    ConeIsolation          = cms.InputTag("hltConeIsolationL3SingleTauMET"),
    InputJets              = cms.InputTag("hltIsolatedL25SingleTauMET"),                             
    IsolatedJets           = cms.InputTag("hltIsolatedL3SingleTauMET"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L3'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(200.),
    NBins                  = cms.int32(50)                            
)



SingleTauMETValidation = cms.Sequence(SingleTauMETPathVal + SingleTauMETL2Val + SingleTauMETL25Val+SingleTauMETL3Val)


