import FWCore.ParameterSet.Config as cms

patTriggerEvent = cms.EDProducer(
  "PATTriggerEventProducer"
, processName        = cms.string( 'HLT' )               # default; change only, if you know exactly, what you are doing!
# , triggerResults     = cms.InputTag( 'TriggerResults' )  # default; change only, if you know exactly, what you are doing!
# , patTriggerProducer = cms.InputTag( 'patTrigger' )      # default; change only, if you know exactly, what you are doing!
# , condGtTag          = cms.InputTag( 'conditionsInEdm' ) # default; change only, if you know exactly, what you are doing!
# , l1GtTag            = cms.InputTag( 'gtDigis' )         # default; change only, if you know exactly, what you are doing!
, patTriggerMatches  = cms.VInputTag(
  )
)
