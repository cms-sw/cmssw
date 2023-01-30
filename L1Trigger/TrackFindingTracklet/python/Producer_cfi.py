import FWCore.ParameterSet.Config as cms
from L1Trigger.TrackTrigger.TrackQualityParams_cfi import *

TrackFindingTrackletProducer_params = cms.PSet (

  InputTag             = cms.InputTag( "l1tTTTracksFromTrackletEmulation", "Level1TTTracks"), #
  InputTagDTC          = cms.InputTag( "TrackerDTCProducer", "StubAccepted"),              #
  LabelTBout           = cms.string  ( "TrackFindingTrackletProducerTBout" ),              #
  LabelDRin            = cms.string  ( "TrackFindingTrackletProducerDRin"  ),              #
  LabelDR              = cms.string  ( "TrackFindingTrackletProducerDR"    ),              #
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
  EnableTruncation     = cms.bool    ( True  ),                                            # enable emulation of truncation for TBout, KF, KFin, lost stubs are filled in BranchLost
  PrintKFDebug         = cms.bool    ( False ),                                            # print end job internal unused MSB
  UseTTStubResiduals   = cms.bool    ( False ),                                            # stub residuals are recalculated from seed parameter and TTStub position
  TrackQualityPSet     = cms.PSet    ( TrackQualityParams ),


)
