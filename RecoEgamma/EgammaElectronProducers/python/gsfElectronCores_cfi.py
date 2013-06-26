import FWCore.ParameterSet.Config as cms

ecalDrivenGsfElectronCores = cms.EDProducer("GsfElectronCoreEcalDrivenProducer",
    gsfPfRecTracks = cms.InputTag("pfTrackElec"),
    gsfTracks = cms.InputTag("electronGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
    useGsfPfRecTracks = cms.bool(True)
)

gsfElectronCores = cms.EDProducer("GsfElectronCoreProducer",
    ecalDrivenGsfElectronCoresTag = cms.InputTag("ecalDrivenGsfElectronCores"),
    pflowGsfElectronCoresTag = cms.InputTag("pfElectronTranslator:pf"),
    gsfPfRecTracks = cms.InputTag("pfTrackElec"),
    gsfTracks = cms.InputTag("electronGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
    useGsfPfRecTracks = cms.bool(True),
    pfSuperClusters = cms.InputTag("pfElectronTranslator:pf"),
    pfSuperClusterTrackMap = cms.InputTag("pfElectronTranslator:pf")
)


