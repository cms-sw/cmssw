import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtTPTriggerMonitor = DQMEDAnalyzer('DTLocalTriggerTask',
    # set static booking (all the detector)
    staticBooking = cms.untracked.bool(True),
    # labels of TM data and 4D segments
    tm_label = cms.untracked.InputTag('twinMuxStage2Digis:PhIn'),
    ros_label = cms.untracked.InputTag('dtunpacker'),
    seg_label = cms.untracked.InputTag('dt4DSegments'),
    minBXTM = cms.untracked.int32(0), # min BX for TM plots
    maxBXTM = cms.untracked.int32(2),  # max BX for TM plots
    process_seg = cms.untracked.bool(False), # if true enables comparisons with reconstructed segments    
    process_tm = cms.untracked.bool(True),  # if true enables TM data analysis
    testPulseMode = cms.untracked.bool(True), #if true enables test pulse mode
    detailedAnalysis = cms.untracked.bool(False), #if true enables detailed analysis plots
    enableTMTheta = cms.untracked.bool(False), # if true enables theta plots for TM
    localrun = cms.untracked.bool(True), # if false access LTC digis
    # number of luminosity blocks to reset the histos
    ResetCycle = cms.untracked.int32(10000)
)


