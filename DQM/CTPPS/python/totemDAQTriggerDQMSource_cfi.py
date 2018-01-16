import FWCore.ParameterSet.Config as cms

totemDAQTriggerDQMSource = DQMStep1Module('TotemDAQTriggerDQMSource',
  tagFEDInfo = cms.InputTag("totemRPRawToDigi", "TrackingStrip"),
  tagTriggerCounters = cms.InputTag("totemTriggerRawToDigi"),

  verbosity = cms.untracked.uint32(0)
)
