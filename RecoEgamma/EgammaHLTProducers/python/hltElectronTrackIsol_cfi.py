import FWCore.ParameterSet.Config as cms

hltEgammaElectronTrackIsolationProducers= cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltL1GsfElectrons" ),
    recoEcalCandidateProducer = cms.InputTag(""),       
    trackProducer = cms.InputTag( "hltL1EgammaRegionalCTFFinalFitWithMaterial" ),
    beamSpotProducer = cms.InputTag("hltOnlineBeamSpot"),
    useGsfTrack = cms.bool(False),
    useSCRefs = cms.bool(False),
    egTrkIsoVetoConeSize = cms.double( 0.0 ),
    egCheckForOtherEleInCone = cms.untracked.bool( False ),
    egTrkIsoPtMin = cms.double( 1.0 ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoZSpan = cms.double( 0.15 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSizeBarrel = cms.double( 0.03 ),
    egTrkIsoVetoConeSizeEndcap = cms.double( 0.03 ),
    egTrkIsoStripBarrel = cms.double( 0.03 ),
    egTrkIsoStripEndcap = cms.double( 0.03 )
)
  
