import FWCore.ParameterSet.Config as cms

logErrorFilter = cms.EDFilter("LogErrorFilter",
                              harvesterTag = cms.InputTag('logErrorHarvester'),
                              atLeastOneError = cms.bool(True),
                              atLeastOneWarning = cms.bool(True),
                              useThresholdsPerKind = cms.bool(False),
                              avoidCategories = cms.vstring('MemoryCheck')
                              )

logErrorSkimFilter = cms.EDFilter("LogErrorFilter",
                              harvesterTag = cms.InputTag('logErrorHarvester'),
                              atLeastOneError = cms.bool(True),
                              atLeastOneWarning = cms.bool(True),
                              useThresholdsPerKind = cms.bool(True),
                              maxErrorKindsPerLumi = cms.uint32(1),    
                              maxWarningKindsPerLumi = cms.uint32(1),    
                              avoidCategories = cms.vstring('MemoryCheck')
                              )

