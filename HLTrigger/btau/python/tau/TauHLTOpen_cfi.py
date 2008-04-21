import FWCore.ParameterSet.Config as cms

hltTauProducer = cms.EDProducer("HLTTauProducer",
    L25TrackIsoJets = cms.InputTag("DUMMY"),
    L3TrackIsoJets = cms.InputTag("DUMMY"),
    SignalCone = cms.double(0.1),
    EcalIsoRMax = cms.double(0.4),
    MatchingCone = cms.double(0.1),
    L2EcalIsoJets = cms.InputTag("DUMMY"),
    PtLeadTk = cms.double(1.0),
    IsolationCone = cms.double(0.5),
    EcalIsoRMin = cms.double(0.13)
)


