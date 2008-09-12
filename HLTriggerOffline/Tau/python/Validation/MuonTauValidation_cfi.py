import FWCore.ParameterSet.Config as cms

MuonTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("TauMCProducer","LeptonicTauMuons"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/MuonTau/Path'),
    L1SeedFilter          = cms.untracked.InputTag("hltL1sMuonTau","","HLT"),
    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterEcalIsolatedTauJetsMuonTau","","HLT"),
    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTau","","HLT"),
    L3SiliconIsolFilter   = cms.untracked.InputTag("DUMMY"),
    MuonFilter            = cms.untracked.InputTag("hltMuonTauIsoL3IsoFiltered","","HLT"),
    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
    NTriggeredTaus        = cms.untracked.uint32(1),
    NTriggeredLeptons     = cms.untracked.uint32(1),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(''),
    LogFileName           = cms.untracked.string(''),
    MatchDeltaRL1         = cms.untracked.double(0.5),
    MatchDeltaRHLT        = cms.untracked.double(0.3)
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



