import FWCore.ParameterSet.Config as cms

subdetFED = cms.EDProducer("SubdetFEDSelector",
    rawInputLabel = cms.InputTag("rawDataCollector"),
    getSiPixel = cms.bool(True),
    getHCAL = cms.bool(True),
    getECAL = cms.bool(False),
    getMuon = cms.bool(False),
    getTrigger = cms.bool(True),
    getSiStrip = cms.bool(False)
)


