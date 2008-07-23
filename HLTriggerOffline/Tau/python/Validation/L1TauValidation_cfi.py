
import FWCore.ParameterSet.Config as cms

L1TauVal = cms.EDAnalyzer("L1TauValidation",
    RefTauCollection      = cms.untracked.InputTag("TauMCProducer","Taus"),
    RefElecCollection   = cms.untracked.InputTag("TauMCProducer","Electrons"),
    RefMuonCollection   = cms.untracked.InputTag("TauMCProducer","Muons"),
                          
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
                                              
    L1MCTauMinDeltaR = cms.double(0.5),
    MCTauHadMinEt = cms.double(15.0),
    MCTauHadMaxAbsEta = cms.double(2.5),

    L1MCElecMinDeltaR = cms.double(0.2),
    MCElecMinEt = cms.double(10.0),
    MCElecMaxAbsEta = cms.double(2.5),

    L1MCMuonMinDeltaR = cms.double(0.2),
    MCMuonMinEt = cms.double(5.0),
    MCMuonMaxAbsEta = cms.double(2.5),

    TriggerTag = cms.string("L1Plots"),
    OutputFileName = cms.string("L1Validation.root")
)


