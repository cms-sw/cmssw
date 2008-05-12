import FWCore.ParameterSet.Config as cms

stripFED = cms.EDFilter("SiStripRegFEDSelector",
    regSeedLabel = cms.InputTag("hltIsolPixelTrackFilter"),
    rawInputLabel = cms.InputTag("rawDataCollector"),
    delta = cms.double(1.0)
)


