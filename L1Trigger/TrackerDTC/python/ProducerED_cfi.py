import FWCore.ParameterSet.Config as cms

TrackerDTCProducer_params = cms.PSet (

  InputTag         = cms.InputTag( "TTStubsFromPhase2TrackerDigis", "StubAccepted" ), # original TTStub selection
  BranchAccepted   = cms.string  ( "StubAccepted" ),                                  # label for prodcut with passed stubs
  BranchLost       = cms.string  ( "StubLost"     ),                                  # label for prodcut with lost stubs
  CheckHistory     = cms.bool    ( False ),                                           # checks if input sample production is configured as current process
  UseHybrid        = cms.bool    ( True  ),                                           # use Hybrid or TMTT as TT algorithm
  EnableTruncation = cms.bool    ( True  )                                            # enable emulation of truncation, lost stubs are filled in BranchLost

)