import FWCore.ParameterSet.Config as cms

hltEgammaElectronTrackIsolationProducers= cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "ctfWithMaterialTracks" ),
    trackProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    egCheckForOtherEleInCone = cms.untracked.bool( False ),
    egTrkIsoPtMin = cms.double( 1.0 ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoZSpan = cms.double( 0.15 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.03 ),
    egTrkIsoStripBarrel = cms.double( 0.03 ),
    egTrkIsoStripEndcap = cms.double( 0.03 )
)

