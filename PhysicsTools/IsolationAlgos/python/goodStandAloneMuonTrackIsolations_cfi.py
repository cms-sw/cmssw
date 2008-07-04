import FWCore.ParameterSet.Config as cms

goodStandAloneMuonTrackIsolations = cms.EDProducer("CandTrackPtIsolationProducer",
    src = cms.InputTag("goodStandAloneMuonTracks"),
    d0Max = cms.double(1000000.0),
    dRMin = cms.double(0.015),
    dRMax = cms.double(0.3),
    elements = cms.InputTag("generalTracks"),
    ptMin = cms.double(1.5),
    dzMax = cms.double(1000000.0)
)


