import FWCore.ParameterSet.Config as cms

hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    throw = cms.bool( False ),
    processName = cms.string( "HLT" ),
    moduleLabelPatternsToMatch = cms.vstring( 'hlt*'),
    moduleLabelPatternsToSkip = cms.vstring(  )
)
