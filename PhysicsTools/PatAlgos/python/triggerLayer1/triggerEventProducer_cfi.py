import FWCore.ParameterSet.Config as cms

patTriggerEvent = cms.EDProducer( "PATTriggerEventProducer",
    processName        = cms.string( 'HLT' ),
    triggerResults     = cms.InputTag( "TriggerResults" ),
    patTriggerProducer = cms.InputTag( "patTrigger" ),
    patTriggerMatches  = cms.VInputTag( "electronTriggerMatchHLTEle15LWL1R"
                                      , "electronTriggerMatchHLTDoubleEle5SWL1R"
                                      , "muonTriggerMatchL1Muon"
                                      , "muonTriggerMatchHLTIsoMu3"
                                      , "muonTriggerMatchHLTMu3"
                                      , "muonTriggerMatchHLTDoubleMu3"
                                      , "tauTriggerMatchHLTDoubleLooseIsoTau15"
                                      )
)
