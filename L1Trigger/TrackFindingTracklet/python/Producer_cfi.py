import FWCore.ParameterSet.Config as cms

TrackFindingTrackletProducer_params = cms.PSet (

  InputTag             = cms.InputTag( "TTTracksFromTrackletEmulation", "Level1TTTracks"), #
  InputTagDTC          = cms.InputTag( "TrackerDTCProducer", "StubAccepted"),              #
  LabelTBout           = cms.string  ( "TrackFindingTrackletProducerTBout" ),              #
  LabelKFin            = cms.string  ( "TrackFindingTrackletProducerKFin"  ),              #
  LabelKF              = cms.string  ( "TrackFindingTrackletProducerKF"    ),              #
  LabelTT              = cms.string  ( "TrackFindingTrackletProducerTT"    ),              #
  LabelAS              = cms.string  ( "TrackFindingTrackletProducerAS"    ),              #
  LabelKFout           = cms.string  ( "TrackFindingTrackletProducerKFout" ),              #
  BranchAcceptedStubs  = cms.string  ( "StubAccepted"  ),                                  #
  BranchAcceptedTracks = cms.string  ( "TrackAccepted" ),                                  #
  BranchLostStubs      = cms.string  ( "StubLost"      ),                                  #
  BranchLostTracks     = cms.string  ( "TrackLost"     ),                                  #
  CheckHistory         = cms.bool    ( False ),                                            # checks if input sample production is configured as current process
  EnableTruncation     = cms.bool    ( True  ),                                            # enable emulation of truncation, lost stubs are filled in BranchLost

)