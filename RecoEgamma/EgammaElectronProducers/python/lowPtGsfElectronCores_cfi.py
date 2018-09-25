import FWCore.ParameterSet.Config as cms

lowPtGsfElectronCores = cms.EDProducer("LowPtGsfElectronCoreProducer",
    gsfPfRecTracks = cms.InputTag("pfTrackElec"),
    gsfTracks = cms.InputTag("electronGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
    useGsfPfRecTracks = cms.bool(True),
    superClusters = cms.InputTag("lowPtGsfElectronSuperClusters")
)
