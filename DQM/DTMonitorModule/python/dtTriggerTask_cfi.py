import FWCore.ParameterSet.Config as cms

triggerMonitor = cms.EDFilter("DTLocalTriggerTask",
    ros_label = cms.untracked.string('dtunpacker'),
    process_ros = cms.untracked.bool(True),
    seg_label = cms.untracked.string('dt4DSegments'),
    process_seg = cms.untracked.bool(True),
    dcc_label = cms.untracked.string('dttpgprod'),
    process_dcc = cms.untracked.bool(True),
    debug = cms.untracked.bool(True),
    localrun = cms.untracked.bool(True)
)


