import FWCore.ParameterSet.Config as cms

dtTriggerBaseMonitor = cms.EDAnalyzer("DTLocalTriggerBaseTask",
    testPulseMode = cms.untracked.bool(False),
    detailedAnalysis = cms.untracked.bool(False),
    targetBXDCC = cms.untracked.int32(0), 
    targetBXDDU = cms.untracked.int32(9),
    bestTrigAccRange = cms.untracked.int32(3),
    processDDU = cms.untracked.bool(True),
    processDCC = cms.untracked.bool(True),
    nTimeBins = cms.untracked.int32(100),
    nLSTimeBin = cms.untracked.int32(15),    
    ResetCycle = cms.untracked.int32(9999),
    inputTagDCC = cms.untracked.InputTag('dttfunpacker'),
    inputTagDDU = cms.untracked.InputTag('dtunpacker'),
    minBXDDU = cms.untracked.int32(0), 
    maxBXDDU = cms.untracked.int32(20),
    minBXDCC = cms.untracked.int32(-2), 
    maxBXDCC = cms.untracked.int32(2)
)


