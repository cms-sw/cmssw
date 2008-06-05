import FWCore.ParameterSet.Config as cms

dtTriggerMonitor = cms.EDFilter("DTLocalTriggerTask",
    # set static booking (all the detector)
    staticBooking = cms.untracked.bool(True),
    ros_label = cms.untracked.string('dtunpacker'),
    seg_label = cms.untracked.string('dt4DSegments'),
    # min BX for DDU plots
    minBXDDU = cms.untracked.int32(0),
    # max BX for DCC plots
    maxBXDCC = cms.untracked.int32(2),
    # min BX for DCC plots
    maxBXDDU = cms.untracked.int32(20),
    # if true enables comparisons with reconstructed segments    
    process_seg = cms.untracked.bool(False),
    # if true enables DDU data analysis
    process_ros = cms.untracked.bool(True),
    # if true enables DCC data analysis
    process_dcc = cms.untracked.bool(False),
    #debug flag
    debug = cms.untracked.bool(False),
    # labels of DDU/DCC data and 4D segments
    dcc_label = cms.untracked.string('muonDTTFDigis'),
    # if true access LTC digis
    localrun = cms.untracked.bool(True),
    # min BX for DCC plots
    minBXDCC = cms.untracked.int32(-2),
    # number of luminosity blocks to reset the histos
    ResetCycle = cms.untracked.int32(10000)
)


