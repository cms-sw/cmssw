import FWCore.ParameterSet.Config as cms

hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(2)
)
