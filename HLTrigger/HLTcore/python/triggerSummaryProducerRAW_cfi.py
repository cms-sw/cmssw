import FWCore.ParameterSet.Config as cms

triggerSummaryProducerRAW = cms.EDFilter("TriggerSummaryProducerRAW",
    processName = cms.string('@')
)


