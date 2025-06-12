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

  BaseShiftx0           = cms.int32(  -1 ), # precision difference in powers of 2 between x0 and inv2R at KF output
  BaseShiftx1           = cms.int32(  -8 ), # precision difference in powers of 2 between x1 and phiT at KF output
  BaseShiftx2           = cms.int32(  -1 ), # precision difference in powers of 2 between x2 and cotTheta at KF output
  BaseShiftx3           = cms.int32(  -1 ), # precision difference in powers of 2 between x3 and zT at KF output

  BaseShiftr0           = cms.int32(  -8 ), # precision difference in powers of 2 between phi residual and phiT
  BaseShiftr1           = cms.int32(   0 ), # precision difference in powers of 2 between z residual and zT

  BaseShiftS00          = cms.int32(  -4 ), # precision difference in powers of 2 between S00 and inv2R x phiT
  BaseShiftS01          = cms.int32( -12 ), # precision difference in powers of 2 between S01 and phiT x phiT
  BaseShiftS12          = cms.int32(   0 ), # precision difference in powers of 2 between S12 and cotTheta x zT
  BaseShiftS13          = cms.int32(  -1 ), # precision difference in powers of 2 between S13 and zT x zT

  BaseShiftR00          = cms.int32(  -5 ), # precision difference in powers of 2 between R00 and phiT x phiT
  BaseShiftR11          = cms.int32(   6 ), # precision difference in powers of 2 between R11 and zT x zT

  BaseShiftInvR00Approx = cms.int32( -30 ),
  BaseShiftInvR11Approx = cms.int32( -41 ),
  BaseShiftInvR00Cor    = cms.int32( -24 ),
  BaseShiftInvR11Cor    = cms.int32( -24 ),
  BaseShiftInvR00       = cms.int32( -30 ), # precision difference in powers of 2 between 1 / R00 and  1 / ( phiT x phiT )
  BaseShiftInvR11       = cms.int32( -41 ), # precision difference in powers of 2 between 1 / R11 and  1 / ( zT x zT )

  BaseShiftS00Shifted   = cms.int32(  -1 ),
  BaseShiftS01Shifted   = cms.int32(  -7 ),
  BaseShiftS12Shifted   = cms.int32(   4 ),
  BaseShiftS13Shifted   = cms.int32(   4 ),

  BaseShiftK00          = cms.int32(  -7 ), # precision difference in powers of 2 between K00 and inv2R / phiT
  BaseShiftK10          = cms.int32( -13 ), # precision difference in powers of 2 between K10 and 1
  BaseShiftK21          = cms.int32( -13 ), # precision difference in powers of 2 between K21 and cotTheta / zT
  BaseShiftK31          = cms.int32( -13 ), # precision difference in powers of 2 between K31 and 1
  
  BaseShiftC00         = cms.int32(   6 ), # precision difference in powers of 2 between C00 and inv2R * inv2R
  BaseShiftC01         = cms.int32(   1 ), # precision difference in powers of 2 between C01 and inv2R * phiT
  BaseShiftC11         = cms.int32(  -6 ), # precision difference in powers of 2 between C11 and phiT * phiT
  BaseShiftC22         = cms.int32(   5 ), # precision difference in powers of 2 between C22 and cotTheta * cotTheta
  BaseShiftC23         = cms.int32(   6 ), # precision difference in powers of 2 between C23 and cotTheta * zT
  BaseShiftC33         = cms.int32(   5 )  # precision difference in powers of 2 between C33 and zT * zT

)
