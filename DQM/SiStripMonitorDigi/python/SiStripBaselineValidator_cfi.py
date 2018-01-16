import FWCore.ParameterSet.Config as cms

SiStripBaselineValidator = DQMStep1Module('SiStripBaselineValidator',
    srcProcessedRawDigi =  cms.InputTag('siStripZeroSuppression','VirginRaw')
)
