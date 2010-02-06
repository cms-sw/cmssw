import FWCore.ParameterSet.Config as cms

triggerSummaryProducerRAW = cms.EDProducer("TriggerSummaryProducerRAW",
    processName = cms.string('@')
)


