import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits
#
pixelMatchElectronsForHLT = cms.EDProducer("EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag("ctfWithMaterialTracksBarrel"),
    GsfTrackProducer = cms.InputTag(""),
    UseGsfTracks = cms.bool(False),   
    BSProducer = cms.InputTag("offlineBeamSpot")
)


