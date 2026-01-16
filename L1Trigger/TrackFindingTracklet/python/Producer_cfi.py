# configuration for hybrid track reconstruction chain emulating EDProducer

import FWCore.ParameterSet.Config as cms

TrackFindingTrackletProducer_params = cms.PSet (

  InputLabelTFP            = cms.string( "ProducerTQ"      ),                  #
  InputLabelTQ             = cms.string( "ProducerKF"      ),                  #
  InputLabelKF             = cms.string( "ProducerDR"      ),                  #
  InputLabelDR             = cms.string( "ProducerTM"      ),                  #
  InputLabelTM             = cms.string( "l1tTTTracksFromTrackletEmulation" ), #
  BranchStubs              = cms.string( "StubAccepted"    ),                  #
  BranchTracks             = cms.string( "TrackAccepted"   ),                  #
  BranchTTTracks           = cms.string( "TTTrackAccepted" ),                  #
  BranchTruncated          = cms.string( "Truncated"       ),                  #
  PrintKFDebug             = cms.bool  ( False ),                              # print end job internal unused MSB

)
