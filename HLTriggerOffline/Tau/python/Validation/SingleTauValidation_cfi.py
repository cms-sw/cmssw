import FWCore.ParameterSet.Config as cms


SingleTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/SingleTau/Path'),
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.0,0.3,0.3,0.3),    #One per filter
    Filter                = cms.untracked.VInputTag(
                                     cms.InputTag("hltL1sSingleTau","","HLT"),
                                     cms.InputTag("hlt1METSingleTau","","HLT"),
                                     cms.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
                                     cms.InputTag("hltFilterL25SingleTau","","HLT"),
                                     cms.InputTag("hltFilterL3SingleTau","","HLT")
                                     ),
    NTriggeredTaus        = cms.untracked.vuint32(1,1,0,1,1,1), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,0,94,94,94),
    LeptonType            = cms.untracked.vint32(0,0,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True)
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


