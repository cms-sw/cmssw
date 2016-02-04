import FWCore.ParameterSet.Config as cms

stripFED = cms.EDProducer("SiStripRegFEDSelector",
    regSeedLabel = cms.InputTag("hltIsolPixelTrackFilter"),
    rawInputLabel = cms.InputTag("rawDataCollector"),
    delta = cms.double(1.0)
)


