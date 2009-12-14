import FWCore.ParameterSet.Config as cms

gsfElectronCores = cms.EDProducer("GsfElectronCoreProducer",
    gsfTracks = cms.InputTag("electronGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
    pfSuperClusters = cms.InputTag("pfElectronTranslator:pf"),
    pfSuperClusterTrackMap = cms.InputTag("pfElectronTranslator:pf")
)


