import FWCore.ParameterSet.Config as cms

logErrorFilter = cms.EDFilter("LogErrorFilter",
                              harvesterTag = cms.InputTag('logErrorHarvester'),
                              atLeastOneError = cms.bool(True),
                              atLeastOneWarning = cms.bool(True),
                              useThresholdsPerKind = cms.bool(False),
                              avoidCategories = cms.vstring()
                              )

logErrorSkimFilter = cms.EDFilter("LogErrorFilter",
                              harvesterTag = cms.InputTag('logErrorHarvester'),
                              atLeastOneError = cms.bool(True),
                              atLeastOneWarning = cms.bool(True),
                              useThresholdsPerKind = cms.bool(True),
                              maxErrorKindsPerLumi = cms.uint32(3),    
                              maxWarningKindsPerLumi = cms.uint32(3),    
                              avoidCategories = cms.vstring()
                              )

