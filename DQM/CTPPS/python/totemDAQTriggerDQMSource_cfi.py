import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
totemDAQTriggerDQMSource = DQMEDAnalyzer('TotemDAQTriggerDQMSource',
  tagFEDInfo = cms.InputTag("totemRPRawToDigi", "TrackingStrip"),
  tagTriggerCounters = cms.InputTag("totemTriggerRawToDigi"),

  verbosity = cms.untracked.uint32(0)
)
