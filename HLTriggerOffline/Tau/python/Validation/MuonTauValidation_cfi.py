import FWCore.ParameterSet.Config as cms


MuonTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("TauMCProducer","LeptonicTauMuons"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/MuonTau/Path'),
    Filter                = cms.untracked.VInputTag(
                                   cms.InputTag("hltL1sMuonTau","","HLT"),
                                   cms.InputTag("hltMuonTauL1Filtered","","HLT"),
                                   cms.InputTag("hltMuonTauIsoL2PreFiltered","","HLT"),
                                   cms.InputTag("hltMuonTauIsoL2IsoFiltered","","HLT"),
                                   cms.InputTag("hltMuonTauIsoL3PreFiltered","","HLT"),
                                   cms.InputTag("hltMuonTauIsoL3IsoFiltered","","HLT"),
                                   cms.InputTag("hltFilterEcalIsolatedTauJetsMuonTau","","HLT"),
                                   cms.InputTag("hltFilterL25MuonTauPtLeadTk","","HLT"),
                                   cms.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTau","","HLT")
                           ),                           
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3),    #One per filter
    NTriggeredTaus        = cms.untracked.vuint32(1,1,0,0,0,0,0,1,1,1), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,1,1,1,1,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,0,0,0,0,0,94,94,94),
    LeptonType            = cms.untracked.vint32(81,81,93,93,93,93,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True),

)

MuonTauL2Val = cms.EDFilter("HLTTauCaloDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/MuonTau/L2'),
    L2InfoAssociationInput = cms.InputTag("hltL2MuonTauIsolationProducer","L2TauIsolationInfoAssociator"),
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    MET                    = cms.InputTag("hltMet"),
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    L2IsolatedJets         = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(50)                            
)


MuonTauL25Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/MuonTau/L25'),
    ConeIsolation          = cms.InputTag("hltPixelTrackConeIsolationMuonTau"),
    InputJets              = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),                             
    IsolatedJets           = cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorMuonTau"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L25'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(50)                            
)

MuonTauValidation = cms.Sequence(MuonTauPathVal + MuonTauL2Val+MuonTauL25Val)



