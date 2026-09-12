# configuration for TrackerDTCSetup
import FWCore.ParameterSet.Config as cms

TrackerDTCSetup_params = cms.PSet (

  # print constants for f/w setting
  Print = cms.PSet (
    Constants    = cms.bool  ( False ),                        # enables printing of board constants
    EncodingBend = cms.bool  ( False ),                        # enables printing of bend encoding
    IDs          = cms.vint32(),                               # dtcs to be printed [0 - 215]; empty means all
    Path         = cms.string( "/heplnw065/tschuh/work/src/dtc-stub-processing/firmware/luts/" ), # path of prints
  ),

  # Parameter specifying CBC
  CBC = cms.PSet (
    NumRow    = cms.int32( 127 ),    # number of rows read out
    NumCol    = cms.int32(   1 ),    # number of coloumns read out
    NumStub   = cms.int32(   3 ),    # number of stubs collected
    NumBX     = cms.int32(   1 ),    # number of events used to collect stubs
    WidthBend = cms.int32(   4 ),    # number of bits used for internal stub bend
    Pitch     = cms.double( 0.009 ), # strip pitch of outer tracker sensors in cm
    Length    = cms.double( 5.025 ), # strip length of outer tracker sensors in cm
  ),

  # Parameter specifying MPA
  MPA = cms.PSet (
    NumRow    = cms.int32( 120 ),     # number of rows read out
    NumCol    = cms.int32(  16 ),     # number of coloumns read out
    NumStub   = cms.int32(   4 ),     # number of stubs collected
    NumBX     = cms.int32(   2 ),     # number of events used to collect stubs
    WidthBend = cms.int32(   3 ),     # number of bits used for internal stub bend
    Pitch     = cms.double( 0.01   ), # pixel pitch of outer tracker sensors in cm
    Length    = cms.double( 0.1467 ), # pixel length of outer tracker sensors in cm
  ),

  # Parameter specifying CIC
  CIC = cms.PSet (
    NumBX      = cms.int32(  8 ), # number of events collected
    NumStub5g  = cms.int32( 16 ), # number of stubs collected for 5 gbps config
    NumStub10g = cms.int32( 35 ), # number of stubs collected for 10 gbps config
    NumFEC     = cms.int32(  8 ), # number of MPAs/CBCs read out
  ),

  # Parameter specifying Sensor Modules
  SensorModule = cms.PSet (
    NumCIC = cms.int32( 2 ), # number of CICs per sensor module
    TiltApproxSlope     = cms.double( 0.884  ), # In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    TiltApproxIntercept = cms.double( 0.507  ), # In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    TiltUncertaintyR    = cms.double( 0.12   ), # In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    Scattering          = cms.double( 0.5    ), # additional radial uncertainty in cm used to calculate stub phi residual uncertainty to take multiple scattering into account
    BendCut             = cms.double( 1.3125 ), # bend uncertainty in pitch units defining stub pt uncertainty
    ClusterWidth        = cms.vdouble( 1.612, 1.469, 1.183, 1.138, 1.225 ), # average ClusterWidths for ("Barrel2S", "BarrelPSFlat", "BarrelPSTilted", "Disk2S", "DiskPS" )
    AddPhiUncertainty   = cms.vdouble( 0.00045, 0.00015, 0.00035, 0.00155, 0.00055 ), # additional phi uncertainties in rad for ("Barrel2S", "BarrelPSFlat", "BarrelPSTilted", "Disk2S", "DiskPS" )
  ),

  # Parameter specifying single DTC board
  DTC = cms.PSet (
    NumLayer  = cms.int32(  4 ), # max number of layer connected to one DTC
    NumModule = cms.int32( 72 ), # max number of sensor modules connected to one DTC board
    Freq  = cms.double( 360.0 ), # DTC clock frequency in MHz
  ),

  # Parameter specifying Tracker Region
  Region = cms.PSet (
    NumDTC = cms.int32( 24 ),            # number of DTC boards used to readout a detector region
    NumTFP = cms.int32( 18 ),            # number of TFP boards used to process a processing region
    MinPt        = cms.double(  1.7   ), # min pt in GeV defining r-phi Region shape
    MaxD0        = cms.double(  0.0   ), # in cm defining r-phi Region shape
    BeamWindowZ  = cms.double( 15.    ), # half lumi region size in cm defining r-z Region shape
    MaxEta       = cms.double(  2.5   ), # defining r-z Region shape in cm
    ChosenRofPhi = cms.double( 55.    ), # critical radius in cm defining r-phi Region shape
    ChosenRofZ   = cms.double( 57.76  ), # critical radius in cm defining r-z Region shape
  ),

  # Parameter specifying system
  System = cms.PSet (
    NumModule        = cms.int32( 13200 ), # total number of modules
    NumRegion        = cms.int32(     9 ), # number of phi slices the outer tracker readout is organized in
    NumOverlap       = cms.int32(     2 ), # number of regions a reconstructable particles may cross
    NumATCASlot      = cms.int32(    12 ), # number of Slots in used ATCA crates
    SlotLimitPS      = cms.int32(     6 ), # slot number changing from PS to 2S
    SlotLimit10gbps  = cms.int32(     3 ), # slot number changing from 10 gbps to 5gbps
    NumBarrelLayer   = cms.int32(     6 ), # number of barrel layer
    NumBarrelLayerPS = cms.int32(     3 ), # number of barrel ps layer
    NumFramesInfra   = cms.int32(     6 ), # needed gap between events of emp-infrastructure firmware
    NumLayers        = cms.int32(     8 ), # number of detector layer a particle may cross
    BField           = cms.double(   3.81120228767395 ), # in T
    SpeedOfLight     = cms.double(   2.99792458       ), # in e8 m/s
    OuterRadius      = cms.double( 112.7              ), # outer radius of outer tracker in cm
    InnerRadius      = cms.double(  21.8              ), # inner radius of outer tracker in cm
    HalfLength       = cms.double( 270.               ), # half length of outer tracker in cm
    FreqLHC          = cms.double(  40.               ), # LHC bunch crossing rate in MHz
  ),

  # Parameter specifying unified Font-End Stubs
  StubFE = cms.PSet (
    BaseBend   = cms.double(  .5 ), # precision of internal stub bend in pitch units
    BaseCol    = cms.double( 1.  ), # precision of internal stub column in pitch units
    BaseRow    = cms.double(  .5 ), # precision of internal stub row in pitch units
  ),

  # Parameter specifying global Stubs
  StubGL = cms.PSet (
    WidthR   = cms.int32( 14 ), # number of bits used for stub r - ChosenRofPhi
    WidthPhi = cms.int32( 19 ), # number of bits used for stub phi w.r.t. phi region centre
    WidthZ   = cms.int32( 15 ), # number of bits used for stub z
  ),

  # Parameter specifying output Stubs
  StubDTC = cms.PSet (
    NumRingsPS    = cms.vint32 ( 11, 11, 8, 8, 8 ), # number of outer PS rings for disk 1, 2, 3, 4, 5
    LayerRs       = cms.vdouble(  24.9316,  37.1777,  52.2656,  68.7598,  86.0156, 108.3105 ), # mean radius of outer tracker barrel layer
    DiskZs        = cms.vdouble( 131.1914, 154.9805, 185.3320, 221.6016, 265.0195           ), # mean z of outer tracker endcap disks
    Disk2SRsSet   = cms.VPSet( # center radius of outer tracker endcap 2S diks strips
      cms.PSet( Disk2SRs = cms.vdouble( 66.4266, 71.4516, 76.2625, 81.2875, 82.9425, 87.9675, 93.8025, 98.8275, 99.8035, 104.8290 ) ), # disk 1
      cms.PSet( Disk2SRs = cms.vdouble( 66.4266, 71.4516, 76.2625, 81.2875, 82.9425, 87.9675, 93.8025, 98.8275, 99.8035, 104.8290 ) ), # disk 2
      cms.PSet( Disk2SRs = cms.vdouble( 63.9978, 69.0028, 74.2625, 79.2875, 81.9437, 86.9687, 92.4795, 97.5045, 99.8035, 104.8290 ) ), # disk 3
      cms.PSet( Disk2SRs = cms.vdouble( 63.9978, 69.0028, 74.2625, 79.2875, 81.9437, 86.9687, 92.4795, 97.5045, 99.8035, 104.8290 ) ), # disk 4
      cms.PSet( Disk2SRs = cms.vdouble( 63.9978, 69.0028, 74.2625, 79.2875, 81.9437, 86.9687, 92.4795, 97.5045, 99.8035, 104.8290 ) )  # disk 5
    ),
    WidthsND      = cms.vint32 (  0,  0,  1,  1 ), # number of bits used for stub negDisk boolean, determined by if in neg. or pos. z region (barrelPS, barrel2S, diskPS, disk2S)
    WidthsR       = cms.vint32 (  7,  7, 11,  6 ), # number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsZ       = cms.vint32 ( 12,  8,  7,  7 ), # number of bits used for stub z w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsPhi     = cms.vint32 ( 14, 17, 14, 14 ), # number of bits used for stub phi w.r.t. region centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsAlpha   = cms.vint32 (  0,  0,  0,  4 ), # number of bits used for stub row number for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsBend    = cms.vint32 (  3,  4,  3,  4 ), # number of bits used for stub bend number for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesR       = cms.vdouble(   7.5,   7.5, 60. ,    0.  ), # range in stub r which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesZ       = cms.vdouble( 240.,  240.,   7.5,    7.5 ), # range in stub z which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesAlpha   = cms.vdouble(   0.,    0.,   0.,  1024.  ), # range in stub row which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    OffsetRDiskPS = cms.double(   7.5 ), # radial offset in cm applied to disk PS stubs
    MinPt         = cms.double(   2.0 ), # in GeV, used to define output stub data format
  ),

  # Parameter specifying unboxing algorithm of 8bx boxcars
  Unbox = cms.PSet (
    WidthAddr = cms.int32( 6 ), # fifo addr width in stub router firmware
    NumNode   = cms.int32( 8 ), # number of parallel worker
  ),

  # Parameter specifying repacking algorithm of 8bx -> 12 bx -> 18bx
  Repack = cms.PSet (
    In  = cms.int32( 2 ),
    Out = cms.int32( 3 ),
  ),

  # Fimrware specific Parameter
  Firmware = cms.PSet (
    EnableTruncation    = cms.bool( True ),
    WidthDSPa           = cms.int32 (  27  ), # width of the 'A' port of an DSP slice
    WidthDSPb           = cms.int32 (  18  ), # width of the 'B' port of an DSP slice
    WidthDSPc           = cms.int32 (  48  ), # width of the 'C' port of an DSP slice
    WidthAddrBRAM36     = cms.int32 (   9  ), # smallest address width of an BRAM36 configured as broadest simple dual port memory
    WidthAddrBRAM18     = cms.int32 (  10  ), # smallest address width of an BRAM18 configured as broadest simple dual port memory
  ),

)
