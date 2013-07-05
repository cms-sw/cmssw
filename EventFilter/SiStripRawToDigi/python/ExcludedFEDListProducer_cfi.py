import FWCore.ParameterSet.Config as cms

SiStripExcludedFEDListProducer = cms.EDProducer(
    'SiStripExcludedFEDListProducer',
    ProductLabel = cms.InputTag("rawDataCollector")
)
