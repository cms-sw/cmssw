import FWCore.ParameterSet.Config as cms

hltTriggerSummaryAOD = cms.EDProducer("TriggerSummaryProducerAOD",
    moduleLabelPatternsToMatch = cms.vstring(
        'hlt*',
        'l1t*'
    ),
    moduleLabelPatternsToSkip = cms.vstring(),
    processName = cms.string('*'),
    throw = cms.bool(False)
)
