import FWCore.ParameterSet.Config as cms

dtTPTriggerMonitor = cms.EDAnalyzer("DTLocalTriggerTask",
    # set static booking (all the detector)
    staticBooking = cms.untracked.bool(True),
    # labels of DDU/DCC data and 4D segments
    dcc_label = cms.untracked.string('dttfunpacker'),
    ros_label = cms.untracked.string('dtunpacker'),
    seg_label = cms.untracked.string('dt4DSegments'),
    minBXDDU = cms.untracked.int32(0),  # min BX for DDU plots
    maxBXDDU = cms.untracked.int32(20), # max BX for DDU plots
    minBXDCC = cms.untracked.int32(-2), # min BX for DCC plots
    maxBXDCC = cms.untracked.int32(2),  # max BX for DCC plots
    process_seg = cms.untracked.bool(False), # if true enables comparisons with reconstructed segments    
    process_ros = cms.untracked.bool(True),  # if true enables DDU data analysis
    process_dcc = cms.untracked.bool(True),  # if true enables DCC data analysis
    testPulseMode = cms.untracked.bool(True), #if true enables test pulse mode
    detailedAnalysis = cms.untracked.bool(False), #if true enables detailed analysis plots
    enableDCCTheta = cms.untracked.bool(False), # if true enables theta plots for DCC
    localrun = cms.untracked.bool(True), # if false access LTC digis
    # number of luminosity blocks to reset the histos
    ResetCycle = cms.untracked.int32(10000)
)


