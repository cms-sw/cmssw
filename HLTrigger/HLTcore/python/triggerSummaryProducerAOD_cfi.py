import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.HLTrigger_EventContent_cff import *
triggerSummaryProducerAOD = cms.EDFilter("TriggerSummaryProducerAOD",
    TriggerSummaryAOD,
    processName = cms.string('@')
)


