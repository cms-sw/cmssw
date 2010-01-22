import FWCore.ParameterSet.Config as cms

patTriggerEvent = cms.EDProducer( "PATTriggerEventProducer",
    processName        = cms.string( 'HLT' ),
    triggerResults     = cms.InputTag( "TriggerResults" ),
    patTriggerProducer = cms.InputTag( "patTrigger" ),
    patTriggerMatches  = cms.VInputTag( "electronTriggerMatchHLTEle15LWL1R"
                                      , "electronTriggerMatchHLTDoubleEle5SWL1R"
                                      , "muonTriggerMatchHLTIsoMu3"
                                      , "muonTriggerMatchHLTMu3"
                                      , "muonTriggerMatchHLTDoubleMu3"
                                      , "tauTriggerMatchHLTDoubleLooseIsoTau15"
                                      )
)
patTriggerEventTest = cms.EDProducer( "PATTriggerEventProducer",
    processName        = cms.string( 'HLT' ),
    triggerResults     = cms.InputTag( "TriggerResults" ),
    patTriggerProducer = cms.InputTag( "patTrigger" ),
    patTriggerMatches  = cms.VInputTag( "electronTriggerTestMatchHLTElectrons"
                                      , "electronTriggerTestMatchHLTFilterEGammas"
                                      , "muonTriggerTestMatchL1Muons"
                                      , "muonTriggerTestMatchL1CollectionMuons"
                                      , "muonTriggerTestMatchNoMuons"
                                      , "jetTriggerTestMatchHLTJet15U"
                                      , "metTriggerTestMatchHLTMET45"
                                      , "metTriggerTestMatchHLTMu3"
                                      )
)
