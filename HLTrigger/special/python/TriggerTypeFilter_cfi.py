import FWCore.ParameterSet.Config as cms

triggerTypeFilter = cms.EDFilter("TriggerTypeFilter",
    TriggerFedId = cms.int32(812),
    InputLabel = cms.string('source'),
    SelectedTriggerType = cms.int32(2)
)


