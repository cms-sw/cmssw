# configuration for TrackTriggerSetup

import FWCore.ParameterSet.Config as cms

TrackFindingTracklet_params = cms.PSet (

  EnableTruncation = cms.bool( True ),

  # Parameter specifying simuation
  Sim = cms.PSet (
    NPar = cms.int32 ( 5 ), # use either 4 or 5 parameter fit in simulation
  ),

  # Parameter specifying outer tracker
  OT = cms.PSet (
    LimitPSBarrel = cms.double ( 125.0 ),                              # barrel layer limit r value to partition into PS and 2S region
    LimitsTiltedR = cms.vdouble(  30.0,  45.0,  60.0 ),                # barrel layer limit r value to partition into tilted and untilted region
    LimitsTiltedZ = cms.vdouble(  15.5,  25.0,  34.3 ),                # barrel layer limit |z| value to partition into tilted and untilted region
    LimitsPSDiksZ = cms.vdouble( 125.0, 150.0, 175.0, 200.0, 250.0 ),  # endcap disk limit |z| value to partition into PS and 2S region
    LimitsPSDiskR = cms.vdouble(  66.4,  66.4,  64.55, 64.55, 64.55 ), # endcap disk limit r value to partition into PS and 2S region
  ),

  # Parameter specifying Input Router
  IR = cms.PSet (
    ChannelsIn = cms.vint32( range(0, 48) ) # vector of DTC id indexed by connected IR module id
  ),

  # Parameter specifying Track Builder
  TB = cms.PSet (
    Freq = cms.double( 240.0 ), # f/w frequency in MHz
    MinZ = cms.double( 120.0 ), # smallest disk stub z position after TrackBuilder in cm
    MaxR = cms.double( 120.0 ), # biggest disk stub r position after TrackBuilder in cm
    InnerRadius = cms.double( 19.6 ), # smallest stub radius after TrackBuilder in cm
    NumSeedTypes = cms.int32( 8 ), # number of seed Types
    NumSeedingLayers = cms.int32( 2 ), # number of layers used to form a seed
    NumLayers = cms.int32( 11 ), # number of layers
    SeedTypes = cms.vstring( "L1L2", "L2L3", "L3L4", "L5L6", "D1D2", "D3D4", "L1D1", "L2D1" ), # seed types used in tracklet algorithm (position gives int value)
    SeedTypesSeedLayers = cms.PSet ( # seeding layers of seed types using default layer id [barrel: 1-6, discs: 11-15]
      L1L2 = cms.vint32(  1,  2 ),
      L2L3 = cms.vint32(  2,  3 ),
      L3L4 = cms.vint32(  3,  4 ),
      L5L6 = cms.vint32(  5,  6 ),
      D1D2 = cms.vint32( 11, 12 ),
      D3D4 = cms.vint32( 13, 14 ),
      L1D1 = cms.vint32(  1, 11 ),
      L2D1 = cms.vint32(  2, 11 )
    ),
    SeedTypesProjectionLayers = cms.PSet ( # layers a seed types can project to using default layer id [barrel: 1-6, discs: 11-15]
      L1L2 = cms.vint32(  3,  4,  5,  6, 11, 12, 13, 14 ),
      L2L3 = cms.vint32(  1,  4,  5,  6, 11, 12, 13, 14 ),
      L3L4 = cms.vint32(  1,  2,  5,  6, 11, 12 ),
      L5L6 = cms.vint32(  1,  2,  3,  4 ),
      D1D2 = cms.vint32(  1,  2, 13, 14, 15 ),
      D3D4 = cms.vint32(  1, 11, 12, 15 ),
      L1D1 = cms.vint32( 12, 13, 14, 15 ),
      L2D1 = cms.vint32(  1, 12, 13, 14 )
    ),
    WidthsR = cms.vint32 ( 7, 7, 12, 12 ), # number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S) at TB output
    WidthStubId  = cms.int32( 10 ), # number of Bits used to represent stubId
    WidthInv2R   = cms.int32( 14 ), # number of Bits used to represent inv2R
    WidthPhi0    = cms.int32( 18 ), # number of Bits used to represent phi0
    WidthZ0      = cms.int32( 11 ), # number of Bits used to represent z0
    WidthCot     = cms.int32( 15 ), # number of Bits used to represent cot
  ),

  # Parameter specifying Track Multiplexer
  TM = cms.PSet (
    # The order here approximately dictates the order in which tracks enter the DR,
    # with DR keeping 1st track to arrive, in case two tracks are duplicate.
    MuxOrder = cms.vstring( "L1L2", "L2L3", "L1D1", "L2D1", "D1D2", "D3D4", "L3L4", "L5L6" ), # seed priority during merge
  ),

  # Parameter specifying Duplicate Removal
  DR = cms.PSet (
    # Next two options enable cheats that replace the tracklet digi output.
    UseDTCStubs          = cms.bool ( False ), # recalculates track parameter and stub residuals from DTC stubs
    UseTTStubs           = cms.bool ( False ), # recalculates track parameter and stub residuals from TT stubs
    NumComparisonModules = cms.int32(    32 ), # number of comparison modules used in each DR node
    MinIdenticalStubs    = cms.int32(     3 ), # min number of shared stubs to identify duplicates
    WidthR               = cms.int32(    14 ), # number of bits used for stub r - ChosenRofPhi
    WidthPhi             = cms.int32(    16 ), # number of bits used for stub phi w.r.t. phi region centre
    WidthZ               = cms.int32(    15 ), # number of bits used for stub z
    WidthDPhi            = cms.int32(     9 ), # number of Bits used to represent stub phi uncertainty in rad
    WidthDZ              = cms.int32(     9 ), # number of Bits used to represent stub z uncertainty in cm
    BaseShiftDPhi        = cms.int32(     1 ), # precision difference in powers of 2 between dPhi and phi
    BaseShiftDZ          = cms.int32(     1 ), # precision difference in powers of 2 between dZ and z
  ),

  # Parmeter specifying KalmanFilter
  KF = cms.PSet (
    UseSimulation = cms.bool ( False ), # simulate KF instead of emulate
    MaxTracks     = cms.int32(    63 ), # max number of tracks a kf worker can process
    NumLayers     = cms.int32(     7 ), # number of layers a fitted track may cross
    MinLayers     = cms.int32(     4 ), # required number of layers to form a track
    BaseShiftPhi  = cms.int32(     2 ), # precision difference in powers of 2 between phi residual and phi position
    BaseShiftZ    = cms.int32(     2 ), # precision difference in powers of 2 between z residual and z position
  ),

  # Parameter specifying Track Quality
  TQ = cms.PSet (
    NumChannel     = cms.int32(   2 ), # number of output channel
    WidthChi21     = cms.int32(   8 ), # Number of bits used to represent chi2rphi
    WidthChi20     = cms.int32(   8 ), # Number of bits used to represent chi2rz
    BaseShiftChi21 = cms.int32(  -3 ), # Base of chi2rphi gets shifted by that power of 2 w.r.t 1
    BaseShiftChi20 = cms.int32(  -3 ), # Base of chi2rz gets shifted by that power of 2 w.r.t 1
    WidthInvV0     = cms.int32(  16 ), # Number of bits used for looked up inverse phi uncertainty squared
    WidthInvV1     = cms.int32(  16 ), # Number of bits used for looked up inverse z uncertainty squared
    WidthMVA       = cms.int32(   3 ), # number of bits used for mva
    BinEdges       = cms.vint32( -14142, -1990, -1126, -523, 0, 523, 1126, 1990, 14142 ) # f/w bin edge integer values to bin mva
  ),

  # Parmeter specifying Track Finding Processor 
  TFP = cms.PSet (
    WidthPhi0  = cms.int32( 12 ), # number of bist used for phi0
    WidthInvR  = cms.int32( 15 ), # number of bist used for invR
    WidthCot   = cms.int32( 16 ), # number of bist used for cot(theta)
    WidthZ0    = cms.int32( 12 ), # number of bist used for z0
    NumChannel = cms.int32(  2 ) # number of output links
  ),

)
