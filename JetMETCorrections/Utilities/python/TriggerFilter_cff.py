import FWCore.ParameterSet.Config as cms
TriggerFilter = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_DiJetAve15U_8E29')
)

skimTriggerBit = cms.Sequence(TriggerFilter)
