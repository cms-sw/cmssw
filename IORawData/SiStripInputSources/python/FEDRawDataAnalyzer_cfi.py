import FWCore.ParameterSet.Config as cms

FedAnalyzer = cms.EDAnalyzer("FEDRawDataAnalyzer",
    ProductLabel = cms.untracked.string('source'),
    ProductInstance = cms.untracked.string(''),
    pause_us = cms.untracked.int32(0)
)


