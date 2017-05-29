import FWCore.ParameterSet.Config as cms

dtTriggerMonitor = cms.EDAnalyzer("DTLocalTriggerTask",
    # set static booking (all the detector)
    staticBooking = cms.untracked.bool(True),
    # labels of DDU/TM data and 4D segments
    tm_label = cms.untracked.string('twinMuxStage2Digis:PhIn'),
    tmTh_label = cms.untracked.InputTag("twinMuxStage2Digis","ThIn"),
    ros_label = cms.untracked.string('dtunpacker'),
    seg_label = cms.untracked.string('dt4DSegments'),
    minBXDDU = cms.untracked.int32(0),  # min BX for DDU plots
    maxBXDDU = cms.untracked.int32(20), # max BX for DDU plots
    maxBXTM = cms.untracked.int32(2),  # max BX for TM plots
    minBXTM = cms.untracked.int32(0), # min BX for TM plots
    process_seg = cms.untracked.bool(False), # if true enables comparisons with reconstructed segments    
    process_ddu = cms.untracked.bool(False),  # if true enables DDU data analysis
    process_tm = cms.untracked.bool(True), # if true enables TM data analysis
    testPulseMode = cms.untracked.bool(False), # if true enables test pluse mode
    detailedAnalysis = cms.untracked.bool(False), #if true enables detailed analysis plots
    enableTMTheta = cms.untracked.bool(True), # if true enables theta plots for TM
    localrun = cms.untracked.bool(True), # if false access LTC digis
    # number of luminosity blocks to reset the histos
    ResetCycle = cms.untracked.int32(10000)
)

#
# Modify for running in run 2 2016 data
#
from Configuration.Eras.Modifier_run2_25ns_specific_cff import run2_25ns_specific
run2_25ns_specific.toModify( dtTriggerMonitor, process_ddu = cms.untracked.bool(False))

