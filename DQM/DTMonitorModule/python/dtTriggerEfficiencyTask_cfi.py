import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtTriggerEfficiencyMonitor = DQMEDAnalyzer('DTTriggerEfficiencyTask',
    # label for muons
    inputTagMuons = cms.untracked.InputTag('muons'),

    SegmArbitration = cms.untracked.string("SegmentAndTrackArbitration"),

    # labels of DDU/TM data and 4D segments
    inputTagTM = cms.untracked.InputTag('dttfDigis'),
#    inputTagTM = cms.untracked.InputTag('dttfDigis'),
    inputTagDDU = cms.untracked.InputTag('muonDTDigis'),
    inputTagSEG = cms.untracked.InputTag('dt4DSegments'),
    inputTagGMT = cms.untracked.InputTag('gtDigis'),
    processDDU = cms.untracked.bool(True),  # Not needed any longer. Switches below for 2016 Eras and onwards
    processTM = cms.untracked.bool(True), # if true enables TM data analysis
    minBXDDU = cms.untracked.int32(7),  # min BX for DDU eff computation
    maxBXDDU = cms.untracked.int32(15), # max BX for DDU eff computation

    checkRPCtriggers = cms.untracked.bool(False), #  Not needed any longer. Swittches below for 2016 Eras and onwards

    nMinHitsPhi = cms.untracked.int32(5),
    phiAccRange = cms.untracked.double(30.),

    detailedAnalysis = cms.untracked.bool(False), #if true enables detailed analysis plots
)

#
# Modify for running in run 2 2016 data
#
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( dtTriggerEfficiencyMonitor, checkRPCtriggers = cms.untracked.bool(False),processDDU = cms.untracked.bool(False))


