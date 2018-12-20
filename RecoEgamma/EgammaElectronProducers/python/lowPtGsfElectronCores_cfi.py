import FWCore.ParameterSet.Config as cms

lowPtGsfElectronCores = cms.EDProducer("LowPtGsfElectronCoreProducer",
    gsfPfRecTracks = cms.InputTag("lowPtGsfElePfGsfTracks"),
    gsfTracks = cms.InputTag("lowPtGsfEleGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
    useGsfPfRecTracks = cms.bool(True),
    superClusters = cms.InputTag("lowPtGsfElectronSuperClusters")
)
