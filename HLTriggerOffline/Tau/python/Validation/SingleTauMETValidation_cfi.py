import FWCore.ParameterSet.Config as cms


SingleTauMETPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/SingleTauMET/Path'),
    Filter                = cms.untracked.VInputTag(
                               cms.InputTag("hltL1sSingleTauMET","","HLT"),
                               cms.InputTag("hlt1METSingleTauMET","","HLT"),
                               cms.InputTag("hltFilterSingleTauMETEcalIsolation","","HLT"),
                               cms.InputTag("hltFilterL25SingleTauMET","","HLT"),
                               cms.InputTag("hltFilterL3SingleTauMET","","HLT")
                           ),
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.0,0.3,0.3,0.3),    #One per filter
    NTriggeredTaus        = cms.untracked.vuint32(1,1,0,1,1,1), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,0,94,94,94),
    LeptonType            = cms.untracked.vint32(0,0,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True)
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


