# configuartion for L1 track Producer

import FWCore.ParameterSet.Config as cms

TrackerTFPProducer_params = cms.PSet (

  InputLabelPP     = cms.string( "ProducerDTC"   ),  #
  InputLabelGP     = cms.string( "ProducerPP"    ),  #
  InputLabelHT     = cms.string( "ProducerGP"    ),  #
  InputLabelCTB    = cms.string( "ProducerHT"    ),  #
  InputLabelKF     = cms.string( "ProducerCTB"   ),  #
  InputLabelDR     = cms.string( "ProducerKF"    ),  #
  InputLabelTQ     = cms.string( "ProducerDR"    ),  #
  InputLabelTFP    = cms.string( "ProducerTQ"    ),  #
  BranchStubs      = cms.string( "StubAccepted"  ),  # branch for prodcut with passed stubs
  BranchTracks     = cms.string( "TrackAccepted" ),  # branch for prodcut with passed tracks
  BranchTTTracks   = cms.string( "TrackAccepted" ),  # branch for prodcut with passed TTTracks
  BranchTruncated  = cms.string( "Truncated"     ),  # branch for truncated prodcuts
  PrintKFDebug     = cms.bool  ( True  )             # print end job internal unused MSB

)
