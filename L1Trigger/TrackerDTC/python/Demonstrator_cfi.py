# configuration for Demonstrator

import FWCore.ParameterSet.Config as cms

TrackerDTCDemonstrator_params = cms.PSet (

  Enable   = cms.bool  (  True ), # enables comparison of s/w with f/w
  IDs      = cms.vint32(       ), # DTC ids [0-215] under test, empty means all beeing tested
  RunTime  = cms.double(    6. ), # runtime in us
  Num8BX   = cms.int32 (    9  ), # number of 8 bx boxcars played in one test
  PathIPBB = cms.string( "/data/tschuh/work/proj/dtc_" ), # path to ipbb proj area
  InputTag = cms.InputTag( "TTStubsFromPhase2TrackerDigis", "StubAccepted" ), # original TTStub selection

)
