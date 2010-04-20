import FWCore.ParameterSet.Config as cms

patTriggerEvent = cms.EDProducer( "PATTriggerEventProducer"
                                , processName        = cms.string( 'HLT' )
#                                 , triggerResults     = cms.InputTag( "TriggerResults" ) # default; change only, if you know exactly, what you are doing!
#                                 , patTriggerProducer = cms.InputTag( "patTrigger" )     # default; change only, if you know exactly, what you are doing!
                                , patTriggerMatches  = cms.VInputTag( "electronTriggerMatchHLTEle15LWL1R"
                                                                    , "electronTriggerMatchHLTDoubleEle5SWL1R"
                                                                    , "muonTriggerMatchL1Muon"
                                                                    , "muonTriggerMatchHLTIsoMu3"
                                                                    , "muonTriggerMatchHLTMu3"
                                                                    , "muonTriggerMatchHLTDoubleMu3"
                                                                    , "tauTriggerMatchHLTDoubleLooseIsoTau15"
                                                                    )
                                )
