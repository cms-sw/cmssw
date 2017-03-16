import FWCore.ParameterSet.Config as cms

dtTriggerEfficiencyMonitor = cms.EDAnalyzer("DTTriggerEfficiencyTask",
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

    checkRPCtriggers = cms.untracked.bool(False), #  Not needed any longer. Switches below for Eras do not work...
    nMinHitsPhi = cms.untracked.int32(5),
    phiAccRange = cms.untracked.double(30.),

    detailedAnalysis = cms.untracked.bool(False), #if true enables detailed analysis plots
)

#
# Modify for running in run 2 2016 data
#
from Configuration.Eras.Modifier_run2_25ns_specific_cff import run2_25ns_specific
run2_25ns_specific.toModify( dtTriggerEfficiencyMonitor, checkRPCtriggers = cms.untracked.bool(False),processDDU = cms.untracked.bool(False))

from Configuration.Eras.Modifier_run2_HI_specific_cff import run2_HI_specific
run2_HI_specific.toModify( dtTriggerEfficiencyMonitor, checkRPCtriggers = cms.untracked.bool(False),processDDU = cms.untracked.bool(False))

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
pA_2016.toModify( dtTriggerEfficiencyMonitor, checkRPCtriggers = cms.untracked.bool(False),processDDU = cms.untracked.bool(False))



