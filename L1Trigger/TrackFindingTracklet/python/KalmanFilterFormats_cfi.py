import FWCore.ParameterSet.Config as cms

HybridKalmanFilterFormats_params = cms.PSet (

  EnableIntegerEmulation = cms.bool( True ), # enables emulation of integer calculations

  WidthR00 = cms.int32( 20 ), # number of bits used to represent R00
  WidthR11 = cms.int32( 20 ), # number of bits used to represent R11
  
  WidthC00 = cms.int32( 20 ), # number of bits used to represent C00
  WidthC01 = cms.int32( 20 ), # number of bits used to represent C01
  WidthC11 = cms.int32( 20 ), # number of bits used to represent C11
  WidthC22 = cms.int32( 20 ), # number of bits used to represent C22
  WidthC23 = cms.int32( 20 ), # number of bits used to represent C23
  WidthC33 = cms.int32( 20 ), # number of bits used to represent C33

# configuration of internal KF variable bases which can be shifted by powers of 2 w.r.t. KF output track parameter
# TrackerTFPProducer_params.PrintKFDebug printouts unused MSB for each variable, so that one could consider decreasing the basseshift by that amount
# numerical instabillity (negative C00, C11, C22, C33) requires smaller baseshifts of related variables (rx, Sxx, Kxx, Rxx, invRxx)
# if a variable overflows an Exception will be thrown and the corresponding baseshift needs to be increased.

  BaseShiftx0           = cms.int32(  -1 ),
  BaseShiftx1           = cms.int32(  -8 ),
  BaseShiftx2           = cms.int32(  -1 ),
  BaseShiftx3           = cms.int32(  -1 ),

  BaseShiftr0           = cms.int32(  -8 ),
  BaseShiftr1           = cms.int32(   0 ),

  BaseShiftS00          = cms.int32(  -4 ),
  BaseShiftS01          = cms.int32( -12 ),
  BaseShiftS12          = cms.int32(   0 ),
  BaseShiftS13          = cms.int32(  -1 ),

  BaseShiftR00          = cms.int32(  -5 ),
  BaseShiftR11          = cms.int32(   6 ),

  BaseShiftInvR00Approx = cms.int32( -30 ),
  BaseShiftInvR11Approx = cms.int32( -41 ),
  BaseShiftInvR00Cor    = cms.int32( -24 ),
  BaseShiftInvR11Cor    = cms.int32( -24 ),
  BaseShiftInvR00       = cms.int32( -30 ),
  BaseShiftInvR11       = cms.int32( -41 ),

  BaseShiftS00Shifted   = cms.int32(  -1 ),
  BaseShiftS01Shifted   = cms.int32(  -7 ),
  BaseShiftS12Shifted   = cms.int32(   4 ),
  BaseShiftS13Shifted   = cms.int32(   4 ),

  BaseShiftK00          = cms.int32(  -7 ),
  BaseShiftK10          = cms.int32( -13 ),
  BaseShiftK21          = cms.int32( -13 ),
  BaseShiftK31          = cms.int32( -13 ),
  
  BaseShiftC00         = cms.int32(   6 ),
  BaseShiftC01         = cms.int32(   1 ),
  BaseShiftC11         = cms.int32(  -6 ),
  BaseShiftC22         = cms.int32(   5 ),
  BaseShiftC23         = cms.int32(   6 ),
  BaseShiftC33         = cms.int32(   5 ),

  BaseShiftr02          = cms.int32(  -2 ),
  BaseShiftr12          = cms.int32(  10 ),
  BaseShiftchi20        = cms.int32( -10 ),
  BaseShiftchi21        = cms.int32( -10 )

)
