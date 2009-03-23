import FWCore.ParameterSet.Config as cms

hcalTimingMonitor = cms.EDAnalyzer("HcalTimingMonitorModule",
    monitorName   = cms.untracked.string('HcalTimingMonitor'),
    prescaleLS    = cms.untracked.int32(1),
    prescaleEvt   = cms.untracked.int32(1),
    l1GtUnpack    = cms.untracked.string('l1GtUnpack'),
    GCTTriggerBit1= cms.untracked.int32(16),
    GCTTriggerBit2= cms.untracked.int32(17),
    GCTTriggerBit3= cms.untracked.int32(18),
    GCTTriggerBit4= cms.untracked.int32(47),
    GCTTriggerBit5= cms.untracked.int32(55),
    CosmicsCorr   = cms.untracked.bool(True),
    Debug         = cms.untracked.bool(True)
)
