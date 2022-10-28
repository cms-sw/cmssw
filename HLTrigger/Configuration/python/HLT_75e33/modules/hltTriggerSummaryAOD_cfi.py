import FWCore.ParameterSet.Config as cms

hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    throw = cms.bool( False ),
    processName = cms.string( "*" ),
    moduleLabelPatternsToMatch = cms.vstring( 'hlt*','l1t*' ),
    moduleLabelPatternsToSkip = cms.vstring(  )
)
