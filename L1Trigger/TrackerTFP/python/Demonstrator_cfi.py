# configuration of Demonstrator.
import FWCore.ParameterSet.Config as cms

TrackTriggerDemonstrator_params = cms.PSet (

  # tmtt
  #LabelIn  = cms.string( "TrackerTFPProducerKFin"          ), #
  #LabelOut = cms.string( "TrackerTFPProducerKF"            ), #
  #DirIPBB  = cms.string( "/heplnw039/tschuh/work/proj/kf/" ), # path to ipbb proj area
  #RunTime  = cms.double( 6.0 )                                # runtime in us

  # hybrid
  LabelIn  = cms.string( "TrackFindingTrackletProducerIRin"  ), #
  LabelOut = cms.string( "TrackFindingTrackletProducerTBout" ), #
  DirIPBB  = cms.string( "/heplnw039/tschuh/work/proj/sim/"  ), # path to ipbb proj area
  RunTime  = cms.double( 8.0 )                                  # runtime in us

)