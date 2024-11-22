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
  EnableTruncation         = cms.bool  ( True  ),                              # enable emulation of truncation for TM, DR, KF, TQ and TFP
  PrintKFDebug             = cms.bool  ( False ),                              # print end job internal unused MSB
  UseTTStubResiduals       = cms.bool  ( True  ),                              # stub residuals and radius are recalculated from seed parameter and TTStub position
  UseTTStubParameters      = cms.bool  ( True  ),                              # track parameter are recalculated from seed TTStub positions
  ApplyNonLinearCorrection = cms.bool  ( True  ),                              # 
  Use5ParameterFit         = cms.bool  ( False ),                              # double precision simulation of 5 parameter fit instead of bit accurate emulation of 4 parameter fit
  UseKFsimmulation         = cms.bool  ( False )                               # simulate KF instead of emulate

)
