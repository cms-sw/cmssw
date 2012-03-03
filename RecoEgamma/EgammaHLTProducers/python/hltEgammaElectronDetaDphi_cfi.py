import FWCore.ParameterSet.Config as cms

hltEgammaElectronDetaDphi = cms.EDProducer( "EgammaHLTElectronDetaDphiProducer",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    useTrackProjectionToEcal = cms.bool(True)
)

