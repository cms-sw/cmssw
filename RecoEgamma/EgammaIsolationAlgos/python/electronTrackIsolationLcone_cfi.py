import FWCore.ParameterSet.Config as cms

electronTrackIsolationLcone = cms.EDProducer("EgammaElectronTkIsolationProducer",
    absolut = cms.bool(True),
    trackProducer = cms.InputTag("generalTracks"),
    intRadius = cms.double(0.04),
    electronProducer = cms.InputTag("gsfElectrons"),
    extRadius = cms.double(0.4),
    ptMin = cms.double(0.7),
    maxVtxDist = cms.double(0.2),
    BeamspotProducer = cms.InputTag("offlineBeamSpot"),
    maxVtxDistXY     = cms.double(9999.0)
)


