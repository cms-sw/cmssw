import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits
# $Id: pixelMatchElectronsForHLT_cfi.py,v 1.5 2012/01/23 12:56:37 sharper Exp $
#
pixelMatchElectronsForHLT = cms.EDProducer("EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag("ctfWithMaterialTracksBarrel"),
    GsfTrackProducer = cms.InputTag(""),
    UseGsfTracks = cms.bool(False),   
    BSProducer = cms.InputTag("offlineBeamSpot")
)


