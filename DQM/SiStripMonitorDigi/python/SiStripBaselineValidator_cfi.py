import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiStripBaselineValidator = DQMEDAnalyzer('SiStripBaselineValidator',
    srcProcessedRawDigi =  cms.InputTag('siStripZeroSuppression','VirginRaw')
)
