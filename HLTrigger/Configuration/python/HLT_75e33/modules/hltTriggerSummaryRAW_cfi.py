import FWCore.ParameterSet.Config as cms


hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
