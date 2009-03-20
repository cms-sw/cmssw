import FWCore.ParameterSet.Config as cms

gsfElectronCores = cms.EDProducer("GsfElectronCoreProducer",
    gsfTracks = cms.InputTag("electronGsfTracks"),
    pfSuperClusters = cms.InputTag("pfElectronTranslator:pf")
)


