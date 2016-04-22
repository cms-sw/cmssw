import FWCore.ParameterSet.Config as cms

totemDAQTriggerDQMSource = cms.EDAnalyzer("TotemDAQTriggerDQMSource",
  tagFEDInfo = cms.InputTag("totemRPRawToDigi", "RP"),
  tagTriggerCounters = cms.InputTag("totemTriggerRawToDigi")
)
