import FWCore.ParameterSet.Config as cms

hltEgammaElectronDetaDphi = cms.EDProducer( "EgammaHLTElectronDetaDphiProducer",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    recoEcalCandidateProducer = cms.InputTag(""),
    useSCRefs = cms.bool(False),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    useTrackProjectionToEcal = cms.bool(True),
    variablesAtVtx = cms.bool(False)                                                
)

