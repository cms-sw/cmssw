import FWCore.ParameterSet.Config as cms

dtTriggerEfficiencyMonitor = cms.EDAnalyzer("DTTriggerEfficiencyTask",
    # label for muons
    inputTagMuons = cms.untracked.InputTag('muons'),

    SegmArbitration = cms.untracked.string("SegmentAndTrackArbitration"),

    # labels of DDU/TM data and 4D segments
    inputTagTM = cms.untracked.InputTag('dttfDigis'),
    inputTagDDU = cms.untracked.InputTag('muonDTDigis'),
    inputTagSEG = cms.untracked.InputTag('dt4DSegments'),
    inputTagGMT = cms.untracked.InputTag('gtDigis'),
    processDDU = cms.untracked.bool(True),  # if true enables DDU data analysis
    processTM = cms.untracked.bool(True), # if true enables TM data analysis
    minBXDDU = cms.untracked.int32(7),  # min BX for DDU eff computation
    maxBXDDU = cms.untracked.int32(15), # max BX for DDU eff computation

    nMinHitsPhi = cms.untracked.int32(5),
    phiAccRange = cms.untracked.double(30.),

    detailedAnalysis = cms.untracked.bool(False), #if true enables detailed analysis plots
)


