import FWCore.ParameterSet.Config as cms

dtTriggerEfficiencyMonitor = cms.EDAnalyzer("DTTriggerEfficiencyTask",
    # labels of DDU/DCC data and 4D segments
    inputTagDCC = cms.untracked.InputTag('dttfDigis'),
    inputTagDDU = cms.untracked.InputTag('muonDTDigis'),
    inputTagSEG = cms.untracked.InputTag('dt4DSegments'),
    inputTagGMT = cms.untracked.InputTag('gtDigis'),
    processDDU = cms.untracked.bool(True),  # if true enables DDU data analysis
    processDCC = cms.untracked.bool(True), # if true enables DCC data analysis
    minBXDDU = cms.untracked.int32(7),  # min BX for DDU eff computation
    maxBXDDU = cms.untracked.int32(15), # max BX for DDU eff computation
    detailedAnalysis = cms.untracked.bool(False), #if true enables detailed analysis plots
)


