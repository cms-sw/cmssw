import FWCore.ParameterSet.Config as cms

hltTriggerType = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)
