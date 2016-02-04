import FWCore.ParameterSet.Config as cms

ecalDrivenGsfElectronCores = cms.EDProducer("GsfElectronCoreEcalDrivenProducer",
    gsfTracks = cms.InputTag("electronGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
)

gsfElectronCores = cms.EDProducer("GsfElectronCoreProducer",
    ecalDrivenGsfElectronCoresTag = cms.InputTag("ecalDrivenGsfElectronCores"),
    gsfTracks = cms.InputTag("electronGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
    pfSuperClusters = cms.InputTag("pfElectronTranslator:pf"),
    pfSuperClusterTrackMap = cms.InputTag("pfElectronTranslator:pf")
)


