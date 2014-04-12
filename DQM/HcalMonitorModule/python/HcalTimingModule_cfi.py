import FWCore.ParameterSet.Config as cms

hcalTimingMonitor = cms.EDAnalyzer("HcalTimingMonitorModule",
    monitorName     = cms.untracked.string('HcalTiming'),
    subSystemFolder = cms.untracked.string('HcalTiming'),
    prescaleLS      = cms.untracked.int32(1),
    prescaleEvt     = cms.untracked.int32(1),
    L1ADataLabel    = cms.untracked.string('l1GtUnpack'),
    GCTTriggerBit1  = cms.untracked.int32(16),
    GCTTriggerBit2  = cms.untracked.int32(17),
    GCTTriggerBit3  = cms.untracked.int32(18),
    GCTTriggerBit4  = cms.untracked.int32(19),
    GCTTriggerBit5  = cms.untracked.int32(47),
    CosmicsCorr     = cms.untracked.bool(True),
    Debug           = cms.untracked.bool(True),
    hbheDigiCollectionTag = cms.InputTag('hcalunpacker'),
    hoDigiCollectionTag = cms.InputTag('hcalunpacker'),
    hfDigiCollectionTag = cms.InputTag('hcalunpacker')
)
