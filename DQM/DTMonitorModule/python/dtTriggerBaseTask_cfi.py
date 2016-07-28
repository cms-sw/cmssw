import FWCore.ParameterSet.Config as cms

dtTriggerBaseMonitor = cms.EDAnalyzer("DTLocalTriggerBaseTask",
    testPulseMode = cms.untracked.bool(False),
    detailedAnalysis = cms.untracked.bool(False),
    targetBXTM = cms.untracked.int32(0), 
    targetBXDDU = cms.untracked.int32(9),
    bestTrigAccRange = cms.untracked.int32(3),
    processDDU = cms.untracked.bool(True),
    processTM = cms.untracked.bool(True),
    nTimeBins = cms.untracked.int32(100),
    nLSTimeBin = cms.untracked.int32(15),    
    ResetCycle = cms.untracked.int32(9999),
    inputTagTM = cms.untracked.InputTag('twinMuxStage2Digis:PhIn'),
    inputTagTMphIn = cms.untracked.InputTag('twinMuxStage2Digis:PhIn'),
    inputTagTMphOut = cms.untracked.InputTag('twinMuxStage2Digis:PhOut'),
    inputTagTMth = cms.untracked.InputTag('twinMuxStage2Digis:ThIn'),
    inputTagDDU = cms.untracked.InputTag('dtunpacker'),
    minBXDDU = cms.untracked.int32(0), 
    maxBXDDU = cms.untracked.int32(20),
    minBXTM = cms.untracked.int32(-2), 
    maxBXTM = cms.untracked.int32(2)
)


