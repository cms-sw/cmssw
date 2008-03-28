import FWCore.ParameterSet.Config as cms

hltElectronTrackIsol = cms.EDFilter("EgammaHLTElectronTrackIsolationProducers",
    egTrkIsoVetoConeSize = cms.double(0.02),
    trackProducer = cms.InputTag("ctfWithMaterialTracks"),
    electronProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    egTrkIsoConeSize = cms.double(0.2),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(0.1)
)


