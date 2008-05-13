import FWCore.ParameterSet.Config as cms

triggerSummaryProducerAOD = cms.EDFilter("TriggerSummaryProducerAOD",
    processName = cms.string('@')
)


