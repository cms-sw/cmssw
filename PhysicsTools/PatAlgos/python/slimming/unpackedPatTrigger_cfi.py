import FWCore.ParameterSet.Config as cms

unpackedPatTrigger = cms.EDProducer(
  "PATTriggerObjectStandAloneUnpacker"
, patTriggerObjectsStandAlone = cms.InputTag( 'selectedPatTrigger' )
, triggerResults              = cms.InputTag( 'TriggerResults::HLT' )
)
# foo bar baz
# SdBbfpi1DMpzT
# cH4XxjnZEInsL
