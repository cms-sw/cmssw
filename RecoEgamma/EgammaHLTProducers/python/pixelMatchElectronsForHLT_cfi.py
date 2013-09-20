import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits
# $Id: pixelMatchElectronsForHLT_cfi.py,v 1.4 2010/02/16 17:08:04 wmtan Exp $
#
pixelMatchElectronsForHLT = cms.EDProducer("EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag("ctfWithMaterialTracksBarrel"),
    GsfTrackProducer = cms.InputTag(""),
    UseGsfTracks = cms.bool(False),   
    BSProducer = cms.InputTag("offlineBeamSpot")
)


