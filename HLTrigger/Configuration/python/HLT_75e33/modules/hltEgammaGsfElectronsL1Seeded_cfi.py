import FWCore.ParameterSet.Config as cms

hltEgammaGsfElectronsL1Seeded = cms.EDProducer("EgammaHLTPixelMatchElectronProducers",
    BSProducer = cms.InputTag("hltOnlineBeamSpot"),
    GsfTrackProducer = cms.InputTag("hltEgammaGsfTracksL1Seeded"),
    TrackProducer = cms.InputTag(""),
    UseGsfTracks = cms.bool(True)
)
