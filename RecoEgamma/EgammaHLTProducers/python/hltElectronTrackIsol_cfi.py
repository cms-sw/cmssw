import FWCore.ParameterSet.Config as cms

hltEgammaElectronTrackIsolationProducers = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    egTrkIsoVetoConeSize = cms.double( 0.03 ),
    trackProducer = cms.InputTag( "ctfWithMaterialTracks" ),
    electronProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoPtMin = cms.double( 1.0 ),
    egTrkIsoZSpan = cms.double( 0.15 ),
    egCheckForOtherEleInCone = cms.untracked.bool( False ),
    egTrkIsoStripBarrel = cms.double( 0.03 ),
    egTrkIsoStripEndcap = cms.double( 0.03 )
)

