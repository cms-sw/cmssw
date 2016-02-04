import FWCore.ParameterSet.Config as cms

triggerSummaryProducerAOD = cms.EDProducer("TriggerSummaryProducerAOD",
    processName = cms.string('@')
)


