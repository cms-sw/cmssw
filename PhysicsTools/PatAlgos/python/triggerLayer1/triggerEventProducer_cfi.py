import FWCore.ParameterSet.Config as cms

patTriggerEvent = cms.EDProducer( "PATTriggerEventProducer",
    processName        = cms.string( 'HLT' ),
    triggerResults     = cms.InputTag( "TriggerResults" ),
    triggerEvent       = cms.InputTag( "hltTriggerSummaryAOD" ),
    patTriggerProducer = cms.InputTag( "patTrigger" ),
    patTriggerMatches  = cms.VInputTag( "electronTriggerMatchHltElectrons"
                                      , "electronTriggerMatchL1Electrons"
                                      , "muonTriggerMatchL1Muons"
                                      , "muonTriggerMatchAll"
                                      , "muonTriggerMatchNone"
                                      , "tauTriggerMatchTriggerTaus"
                                      )
)

