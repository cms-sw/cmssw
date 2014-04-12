import FWCore.ParameterSet.Config as cms

ecalFED = cms.EDProducer("ECALRegFEDSelector",
    regSeedLabel = cms.InputTag("hltPixelIsolTrackFilter"),
    rawInputLabel = cms.InputTag("rawDataCollector"),
    delta = cms.double(1.0)
)


