import FWCore.ParameterSet.Config as cms

patTriggerEvent = cms.EDProducer( "PATTriggerEventProducer"
, processName        = cms.string( 'HLT' )              # default; change only, if you know exactly, what you are doing!
# , triggerResults     = cms.InputTag( "TriggerResults" ) # default; change only, if you know exactly, what you are doing!
# , patTriggerProducer = cms.InputTag( "patTrigger" )     # default; change only, if you know exactly, what you are doing!
# , l1GtTag            = cms.InputTag( 'gtDigis' )        # default; change only, if you know exactly, what you are doing!
, patTriggerMatches  = cms.VInputTag(
    "electronTriggerMatchHLTEle15LWL1R"
  , "electronTriggerMatchHLTDoubleEle5SWL1R"
  , "muonTriggerMatchL1Muon"
  , "muonTriggerMatchHLTIsoMu3"
  , "muonTriggerMatchHLTMu3"
  , "muonTriggerMatchHLTDoubleMu3"
  , "tauTriggerMatchHLTDoubleLooseIsoTau15"
  )
)
