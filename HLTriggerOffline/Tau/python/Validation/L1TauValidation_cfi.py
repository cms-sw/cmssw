
import FWCore.ParameterSet.Config as cms

L1TauVal = cms.EDAnalyzer("HLTTauL1DQMOfflineSource",
    RefTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    RefElecCollection   = cms.untracked.InputTag("TauMCProducer","LeptonicTauElectrons"),
    RefMuonCollection   = cms.untracked.InputTag("TauMCProducer","LeptonicTauMuons"),
                          
    # L1extra reading
    L1extraCenJetSource = cms.InputTag("hltL1extraParticles","Central"),
    L1extraForJetSource = cms.InputTag("hltL1extraParticles","Forward"),
    L1extraTauJetSource = cms.InputTag("hltL1extraParticles","Tau"),
    L1extraIsoEgammaSource = cms.InputTag("hltL1extraParticles","Isolated"),
    L1extraNonIsoEgammaSource = cms.InputTag("hltL1extraParticles","NonIsolated"),
    L1extraMuonSource = cms.InputTag("hltL1extraParticles"),
    L1extraMETSource = cms.InputTag("hltL1extraParticles"),
    SingleTauThreshold = cms.double(80.0),
    SingleTauMETThresholds = cms.vdouble(30.0, 30.0),
    DoubleTauThreshold = cms.double(40.0),
    MuTauThresholds = cms.vdouble(5.0, 20.0),
    IsoEgTauThresholds = cms.vdouble(10.0, 20.0),                       
                                              
    L1RefTauMinDeltaR = cms.double(0.5),
    RefTauHadMinEt = cms.double(10.0),
    RefTauHadMaxAbsEta = cms.double(2.5),

    L1RefElecMinDeltaR = cms.double(0.5),
    RefElecMinEt = cms.double(5.0),
    RefElecMaxAbsEta = cms.double(2.5),

    L1RefMuonMinDeltaR = cms.double(0.5),
    RefMuonMinEt = cms.double(3.0),
    RefMuonMaxAbsEta = cms.double(2.5),

    TriggerTag = cms.string("HLT/HLTTAU/L1"),
    OutputFileName = cms.string('')
)


