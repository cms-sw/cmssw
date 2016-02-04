import FWCore.ParameterSet.Config as cms

logErrorFilter = cms.EDFilter("LogErrorFilter",
                              harvesterTag = cms.InputTag('logErrorHarvester'),
                              atLeastOneError = cms.bool(True),
                              atLeastOneWarning = cms.bool(True),
                              avoidCategories = cms.vstring()
                              )

