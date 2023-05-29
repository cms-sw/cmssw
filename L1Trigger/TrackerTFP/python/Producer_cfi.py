import FWCore.ParameterSet.Config as cms

TrackerTFPProducer_params = cms.PSet (

  LabelDTC         = cms.string( "TrackerDTCProducer"       ), #
  LabelGP          = cms.string( "TrackerTFPProducerGP"     ), #
  LabelHT          = cms.string( "TrackerTFPProducerHT"     ), #
  LabelMHT         = cms.string( "TrackerTFPProducerMHT"    ), #
  LabelZHT         = cms.string( "TrackerTFPProducerZHT"    ), #
  LabelZHTout      = cms.string( "TrackerTFPProducerZHTout" ), #
  LabelKFin        = cms.string( "TrackerTFPProducerKFin"   ), #
  LabelKF          = cms.string( "TrackerTFPProducerKF"     ), #
  LabelDR          = cms.string( "TrackerTFPProducerDR"     ), #
  LabelTT          = cms.string( "TrackerTFPProducerTT"     ), #
  LabelAS          = cms.string( "TrackerTFPProducerAS"     ), #
  BranchAcceptedStubs  = cms.string( "StubAccepted"  ),        # branch for prodcut with passed stubs
  BranchAcceptedTracks = cms.string( "TrackAccepted" ),        # branch for prodcut with passed tracks
  BranchLostStubs      = cms.string( "StubLost"      ),        # branch for prodcut with lost stubs
  BranchLostTracks     = cms.string( "TracksLost"    ),        # branch for prodcut with lost tracks
  CheckHistory     = cms.bool  ( False ),                      # checks if input sample production is configured as current process
  EnableTruncation = cms.bool  ( True  ),                      # enable emulation of truncation, lost stubs are filled in BranchLost
  PrintKFDebug     = cms.bool  ( False )                       # print end job internal unused MSB

)