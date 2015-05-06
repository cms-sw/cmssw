import FWCore.ParameterSet.Config as cms

SiStripBaselineValidator = cms.EDAnalyzer("SiStripBaselineValidator",
    srcProcessedRawDigi =  cms.InputTag('siStripZeroSuppression','VirginRaw')
)
