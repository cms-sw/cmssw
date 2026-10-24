# configuration for ProducerDTC
import FWCore.ParameterSet.Config as cms

ProducerDTC_params = cms.PSet (

  InputTag = cms.InputTag( "TTStubsFromPhase2TrackerDigis", "StubAccepted" ), # original TTStub selection
  Branch   = cms.string  ( "StubAccepted" ),                                  # label for prodcut with passed stubs

)
