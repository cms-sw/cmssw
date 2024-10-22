import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtTriggerEfficiencyMonitor = DQMEDAnalyzer('DTTriggerEfficiencyTask',
    # label for muons
    inputTagMuons = cms.untracked.InputTag('muons'),

    SegmArbitration = cms.untracked.string("SegmentAndTrackArbitration"),

    # labels of TM data and 4D segments
    inputTagTM = cms.untracked.InputTag('dttfDigis'),
    inputTagSEG = cms.untracked.InputTag('dt4DSegments'),
    inputTagGMT = cms.untracked.InputTag('gtDigis'),
    processTM = cms.untracked.bool(True), # if true enables TM data analysis

    checkRPCtriggers = cms.untracked.bool(False), #  Not needed any longer. Swittches below for 2016 Eras and onwards

    nMinHitsPhi = cms.untracked.int32(5),
    phiAccRange = cms.untracked.double(30.),

    detailedAnalysis = cms.untracked.bool(False), #if true enables detailed analysis plots
)

#
# Modify for running in run 2 2016 data
#
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify( dtTriggerEfficiencyMonitor, inputTagTM = 'twinMuxStage2Digis:PhIn')

