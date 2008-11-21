import FWCore.ParameterSet.Config as cms

DoubleTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/DoubleTau/Path'),
    Filter                = cms.untracked.VInputTag(
                                     cms.InputTag("hltL1sDoubleTau","","HLT"), 
                                     cms.InputTag("hltFilterDoubleTauEcalIsolation","","HLT"),
                                     cms.InputTag("hltFilterL25PixelTauPtLeadTk","","HLT"),
                                     cms.InputTag("hltFilterL25PixelTau","","HLT")
                                     ),
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3),    #One per filter
    NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,94,94,94),
    LeptonType            = cms.untracked.vint32(0,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True),
)

DoubleTauL2Val = cms.EDFilter("HLTTauCaloDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/DoubleTau/L2'),
    L2InfoAssociationInput = cms.InputTag("hltL2DoubleTauIsolationProducer","L2TauIsolationInfoAssociator"),
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    MET                    = cms.InputTag("hltMet"),
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    L2IsolatedJets         = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(50)                            
)


DoubleTauL25Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/DoubleTau/L25'),
    ConeIsolation          = cms.InputTag("hltConeIsolationL25PixelTauIsolated"),
    InputJets              = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),                             
    IsolatedJets           = cms.InputTag("hltIsolatedL25PixelTau"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L25'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(50)                            
)


DoubleTauValidation = cms.Sequence(DoubleTauPathVal + DoubleTauL2Val + DoubleTauL25Val)

