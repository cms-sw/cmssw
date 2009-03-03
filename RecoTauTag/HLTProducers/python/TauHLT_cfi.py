import FWCore.ParameterSet.Config as cms

TauOpenHLT = cms.EDProducer("DQMTauProducer",
    TrackIsoJets = cms.InputTag("hltL25TauConeIsolation"),
        SignalCone = cms.double(0.15),
    MatchingCone = cms.double(0.2),
        IsolationCone = cms.double(0.5),
     MinPtTracks = cms.double(1.)
)
