
import FWCore.ParameterSet.Config as cms

LaserAlignmentConstants = cms.untracked.VPSet( 

  # all beamsplitter kinks (beam0, ..., beam7) in radians
  # the global offsets are syst. offsets observed in the lab measurements wrt. data, ask Bruno..
  cms.PSet(
    PSetName = cms.string( "BeamsplitterKinks" ),
    LASTecPlusRing4BsKinks  = cms.vdouble( -0.00140, -0.00080,  0.00040, -0.00126,  0.00016,  0.00007, -0.00063,  0.00056 ),
    LASTecPlusRing6BsKinks  = cms.vdouble( -0.00253, -0.00027, -0.00207, -0.00120, -0.00198,  0.00082,  0.00069,  0.00001 ),
    TecPlusGlobalOffset = cms.double( 0.0007 ), # global syst. offset added to all kinks in TEC+
    LASTecMinusRing4BsKinks = cms.vdouble(  0.00101,  0.00035, -0.00212,  0.00015,  0.00121, -0.00278,  0.00031, -0.00140 ),
    LASTecMinusRing6BsKinks = cms.vdouble( -0.00047,  0.00036, -0.00235, -0.00043,  0.00025, -0.00159, -0.00258, -0.00048 ),
    TecMinusGlobalOffset = cms.double( 0.0 ),# global syst. offset added to all kinks in TEC-
    LASAlignmentTubeBsKinks = cms.vdouble(  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000 ) # yet unknown
  ),

  # the beam radii in mm
  cms.PSet(
    PSetName = cms.string( "Radii" ),
    LASTecRadius = cms.vdouble( 564., 840. ),
    LASAtRadius  = cms.double( 564. )
  ),

  # z positions in mm
  cms.PSet(
    PSetName = cms.string( "ZPositions" ),
    LASTecZPositions  = cms.vdouble( 1322.5,  1462.5,  1602.5,  1742.5,  1882.5,  2057.5,  2247.5,  2452.5,  2667.5 ),
    LASTibZPositions = cms.vdouble(  620., 380., 180., -100., -340., -540 ),
    LASTobZPositions = cms.vdouble( 1040., 580., 220., -140., -500., -860 ),
    LASTecBeamSplitterZPosition  = cms.double( 2057.5 ),
    LASAtBeamsplitterZPosition = cms.double( 1123. )
  )

)

