import FWCore.ParameterSet.Config as cms

gedGsfElectronCores = cms.EDProducer("GEDGsfElectronCoreProducer",
    GEDEMUnbiased = cms.InputTag("particleFlowEGamma"),
    gsfTracks = cms.InputTag("electronGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
)

