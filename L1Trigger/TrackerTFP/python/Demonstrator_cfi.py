# configuration of TrackTriggerDemonstrator.
import FWCore.ParameterSet.Config as cms

# these parameters a for ModelSim runs of FW
TrackTriggerDemonstrator_params = cms.PSet (

  LabelIn  = cms.string( "ProducerCTB"              ),           #
  LabelOut = cms.string( "ProducerKF"               ),           #
  DirIPBB  = cms.string( "/heplnw039/tschuh/work/proj/ctbkf/" ), # path to ipbb proj area
  RunTime  = cms.double( 6.50 ),                                 # runtime in us

  #LinkMappingIn  = cms.vint32( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 ),
  #LinkMappingOut = cms.vint32( 52, 53, 54, 55, 56, 57, 58, 59, 60 )
  LinkMappingIn  = cms.vint32(),
  LinkMappingOut = cms.vint32()

)
