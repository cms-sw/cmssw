import FWCore.ParameterSet.Config as cms

TrackTrigger_params = cms.PSet (

  fromDD4hep = cms.bool(False),

  # Parameter to check if configured Tracker Geometry is supported
  # this refers to files included by Configuration/Geometry/python/GeometryExtended*_cff.py
  UnSupportedGeometry = cms.PSet (
    XMLLabel    = cms.string ("geomXMLFiles"                             ), # label of ESProducer/ESSource
    XMLPath     = cms.string ("Geometry/TrackerCommonData/data/PhaseII/" ), # compared path
    XMLFile     = cms.string ("tracker.xml"                              ), # compared filen ame
    XMLVersions = cms.vstring()  # list of unsupported versions
  ),

  # Parameter to check if Process History is consistent with process configuration
  ProcessHistory = cms.PSet (
    GeometryConfiguration = cms.string( "XMLIdealGeometryESSource@"                    ), # label of compared GeometryConfiguration
    TTStubAlgorithm       = cms.string( "TTStubAlgorithm_official_Phase2TrackerDigi_@" )  # label of compared TTStubAlgorithm
  ),

  # Common track finding parameter
  TrackFinding = cms.PSet (
    BeamWindowZ      = cms.double( 15. ), # half lumi region size in cm
    MatchedLayers    = cms.int32 (  4  ), # required number of layers a found track has to have in common with a TP to consider it matched to it
    MatchedLayersPS  = cms.int32 (  0  ), # required number of ps layers a found track has to have in common with a TP to consider it matched to it
    UnMatchedStubs   = cms.int32 (  1  ), # allowed number of stubs a found track may have not in common with its matched TP
    UnMatchedStubsPS = cms.int32 (  0  ), # allowed number of PS stubs a found track may have not in common with its matched TP
    Scattering       = cms.double( 0.131283 ) # additional radial uncertainty in cm used to calculate stub phi residual uncertainty to take multiple scattering into account
  ),

  # TMTT specific parameter
  TMTT = cms.PSet (
    MinPt            = cms.double(  3.   ), # cut on stub in GeV, also defines region overlap shape
    MaxEta           = cms.double(  2.4  ), # cut on stub eta
    ChosenRofPhi     = cms.double( 67.24 ), # critical radius defining region overlap shape in cm
    NumLayers        = cms.int32 (  7    ), # number of detector layers a reconstructbale particle may cross, reduced to 7, 8th layer almost never corssed
    WidthR           = cms.int32 ( 12    ), # number of bits used for stub r - ChosenRofPhi
    WidthPhi         = cms.int32 ( 15    ), # number of bits used for stub phi w.r.t. phi region centre
    WidthZ           = cms.int32 ( 14    )  # number of bits used for stub z
  ),

  # Hybrid specific parameter
  Hybrid = cms.PSet (
    MinPtStub    = cms.double(  2.0  ),                        # cut on stub pt in GeV, also defines region overlap shape
    MinPtCand    = cms.double(  1.34 ),                        # cut on candidate pt in GeV
    MaxEta       = cms.double(  2.5  ),                        # cut on stub eta
    ChosenRofPhi = cms.double( 55.   ),                        # critical radius defining region overlap shape in cm
    NumLayers    = cms.int32 (  4    ),                        # max number of detector layer connected to one DTC
    NumRingsPS   = cms.vint32 ( 11, 11, 8, 8, 8 ),             # number of outer PS rings for disk 1, 2, 3, 4, 5
    WidthsR      = cms.vint32 (   7,     7,    12,      7   ), # number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsZ      = cms.vint32 (  12,     8,     7,      7   ), # number of bits used for stub z w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsPhi    = cms.vint32 (  14,    17,    14,     14   ), # number of bits used for stub phi w.r.t. region centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsAlpha  = cms.vint32 (   0,     0,     0,      4   ), # number of bits used for stub row number for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsBend   = cms.vint32 (   3,     4,     3,      4   ), # number of bits used for stub bend number for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesR      = cms.vdouble(   7.5,   7.5, 120. ,    0.  ), # range in stub r which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesZ      = cms.vdouble( 240.,  240.,    7.5,    7.5 ), # range in stub z which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesAlpha  = cms.vdouble(   0.,    0.,    0.,  2048.  ), # range in stub row which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    LayerRs      = cms.vdouble(  24.9316,  37.1777,  52.2656,  68.7598,  86.0156, 108.3105 ), # mean radius of outer tracker barrel layer
    DiskZs       = cms.vdouble( 131.1914, 154.9805, 185.3320, 221.6016, 265.0195           ), # mean z of outer tracker endcap disks
    Disk2SRsSet  = cms.VPSet(                                                                 # center radius of outer tracker endcap 2S diks strips
      cms.PSet( Disk2SRs = cms.vdouble( 66.4391, 71.4391, 76.2750, 81.2750, 82.9550, 87.9550, 93.8150, 98.8150, 99.8160, 104.8160 ) ), # disk 1
      cms.PSet( Disk2SRs = cms.vdouble( 66.4391, 71.4391, 76.2750, 81.2750, 82.9550, 87.9550, 93.8150, 98.8150, 99.8160, 104.8160 ) ), # disk 2
      cms.PSet( Disk2SRs = cms.vdouble( 63.9903, 68.9903, 74.2750, 79.2750, 81.9562, 86.9562, 92.4920, 97.4920, 99.8160, 104.8160 ) ), # disk 3
      cms.PSet( Disk2SRs = cms.vdouble( 63.9903, 68.9903, 74.2750, 79.2750, 81.9562, 86.9562, 92.4920, 97.4920, 99.8160, 104.8160 ) ), # disk 4
      cms.PSet( Disk2SRs = cms.vdouble( 63.9903, 68.9903, 74.2750, 79.2750, 81.9562, 86.9562, 92.4920, 97.4920, 99.8160, 104.8160 ) )  # disk 5
    ),
    InnerRadius = cms.double( 19.6 ), # smallest stub radius after TrackBuilder in cm
  ),

  # Parameter specifying TrackingParticle used for Efficiency measurements
  TrackingParticle = cms.PSet (
    MinPt         = cms.double(  2.  ), # pt cut in GeV
    MaxEta        = cms.double(  2.4 ), # eta cut
    MaxVertR      = cms.double(  1.  ), # cut on vertex pos r in cm
    MaxVertZ      = cms.double( 30.  ), # cut on vertex pos z in cm
    MaxD0         = cms.double(  5.  ), # cut on impact parameter in cm
    MinLayers     = cms.int32 (  4   ), # required number of associated layers to a TP to consider it reconstruct-able and to match it with TTTrack
    MinLayersPS   = cms.int32 (  0   ), # required number of associated ps layers to a TP to consider it reconstruct-able
    MaxBadStubs2S = cms.int32 (  1   ), # max number of unassociated 2S stubs allowed to still associate TTTrack with TP
    MaxBadStubsPS = cms.int32 (  0   )  # max number of unassociated PS stubs allowed to still associate TTTrack with TP
  ),

  # Fimrware specific Parameter
  Firmware = cms.PSet (
    WidthDSPa           = cms.int32(   27                ), # width of the 'A' port of an DSP slice
    WidthDSPb           = cms.int32(   18                ), # width of the 'B' port of an DSP slice
    WidthDSPc           = cms.int32(   48                ), # width of the 'C' port of an DSP slice
    WidthAddrBRAM36     = cms.int32(    9                ), # smallest address width of an BRAM36 configured as broadest simple dual port memory
    WidthAddrBRAM18     = cms.int32(   10                ), # smallest address width of an BRAM18 configured as broadest simple dual port memory
    NumFramesInfra      = cms.int32 (   6                ), # needed gap between events of emp-infrastructure firmware
    FreqLHC             = cms.double(  40.               ), # LHC bunch crossing rate in MHz
    FreqBE              = cms.double( 360.               ), # processing Frequency of DTC, KF & TFP in MHz, has to be integer multiple of FreqLHC
    TMP_FE              = cms.int32 (   8                ), # number of events collected in front-end
    TMP_TFP             = cms.int32 (  18                ), # time multiplexed period of track finding processor
    SpeedOfLight        = cms.double(   2.99792458       ), # in e8 m/s
    BField              = cms.double(   3.81120228767395 ), # in T
    BFieldError         = cms.double(   1.e-6            ), # accepted difference to EventSetup in T
    OuterRadius         = cms.double( 112.7              ), # outer radius of outer tracker in cm
    InnerRadius         = cms.double(  21.8              ), # inner radius of outer tracker in cm
    HalfLength          = cms.double( 270.               ), # half length of outer tracker in cm
    TiltApproxSlope     = cms.double(   0.884            ), # In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    TiltApproxIntercept = cms.double(   0.507            ), # In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    TiltUncertaintyR    = cms.double(   0.12             ), # In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    MindPhi             = cms.double(   0.0001           ), # minimum representable stub phi uncertainty * sqrt(12) + additional terms in rad
    MaxdPhi             = cms.double(   0.02             ), # maximum representable stub phi uncertainty * sqrt(12) + additional terms in rad
    MindZ               = cms.double(   0.1              ), # minimum representable stub z uncertainty * sqrt(12) + additional terms in cm
    MaxdZ               = cms.double(  30.               ), # maximum representable stub z uncertainty * sqrt(12) + additional terms in cm
    Pitch2S             = cms.double(   0.009            ), # strip pitch of outer tracker sensors in cm
    PitchPS             = cms.double(   0.01             ), # pixel pitch of outer tracker sensors in cm
    Length2S            = cms.double(   5.025            ), # strip length of outer tracker sensors in cm
    LengthPS            = cms.double(   0.1467           ), # pixel length of outer tracker sensors in cm
    TiltedLayerLimitsZ  = cms.vdouble( 15.5, 24.9, 34.3, -1., -1., -1. ), # barrel layer limit |z| value to partition into tilted and untilted region
    PSDiskLimitsR       = cms.vdouble( 66.4, 66.4, 64.55, 64.55, 64.55 ), # endcap disk limit r value to partition into PS and 2S region
  ),

  # Parmeter specifying front-end
  FrontEnd = cms.PSet (
    WidthBend      = cms.int32 (  6      ), # number of bits used for internal stub bend
    WidthCol       = cms.int32 (  5      ), # number of bits used for internal stub column
    WidthRow       = cms.int32 ( 11      ), # number of bits used for internal stub row
    BaseBend       = cms.double(   .25   ), # precision of internal stub bend in pitch units
    BaseCol        = cms.double(  1.     ), # precision of internal stub column in pitch units
    BaseRow        = cms.double(   .5    ), # precision of internal stub row in pitch units
    BaseWindowSize = cms.double(   .5    ), # precision of window sizes in pitch units
    BendCut        = cms.double(  1.3125 )  # used stub bend uncertainty in pitch units
  ),

  # Parmeter specifying DTC 
  DTC = cms.PSet (
    NumRegions            = cms.int32(  9 ), # number of phi slices the outer tracker readout is organized in
    NumOverlappingRegions = cms.int32(  2 ), # number of regions a reconstructable particles may cross
    NumATCASlots          = cms.int32( 12 ), # number of Slots in used ATCA crates
    NumDTCsPerRegion      = cms.int32( 24 ), # number of DTC boards used to readout a detector region, likely constructed to be an integerer multiple of NumSlots_
    NumModulesPerDTC      = cms.int32( 72 ), # max number of sensor modules connected to one DTC board
    NumRoutingBlocks      = cms.int32(  2 ), # number of systiloic arrays in stub router firmware
    DepthMemory           = cms.int32( 64 ), # fifo depth in stub router firmware
    WidthRowLUT           = cms.int32(  4 ), # number of row bits used in look up table
    WidthInv2R            = cms.int32(  9 ), # number of bits used for stub inv2R. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
    OffsetDetIdDSV        = cms.int32(  1 ), # tk layout det id minus DetSetVec->detId
    OffsetDetIdTP         = cms.int32( -1 ), # tk layout det id minus TrackerTopology lower det id
    OffsetLayerDisks      = cms.int32( 10 ), # offset in layer ids between barrel layer and endcap disks
    OffsetLayerId         = cms.int32(  1 ), # offset between 0 and smallest layer id (barrel layer 1)
    NumBarrelLayer        = cms.int32(  6 ), #
    SlotLimitPS           = cms.int32(  6 ), # slot number changing from PS to 2S
    SlotLimit10gbps       = cms.int32(  3 )  # slot number changing from 10 gbps to 5gbps
  ),

  # Parmeter specifying TFP 
  TFP = cms.PSet (
    WidthPhi0  = cms.int32( 12 ), # number of bist used for phi0
    WidthInv2R = cms.int32( 15 ), # number of bist used for inv2R
    WidthCot   = cms.int32( 16 ), # number of bist used for cot(theta)
    WidthZ0    = cms.int32( 12 ), # number of bist used for z0
    NumChannel = cms.int32(  2 )  # number of output links
  ),

  # Parmeter specifying GeometricProcessor
  GeometricProcessor = cms.PSet (
    NumSectorsPhi = cms.int32 (   2  ), # number of phi sectors used in hough transform
    ChosenRofZ    = cms.double(  50. ), # critical radius defining r-z sector shape in cm
    RangeChiZ     = cms.double( 160. ), # range of stub z residual w.r.t. sector center which needs to be covered
    DepthMemory   = cms.int32 (  64  ), # fifo depth in stub router firmware
    #BoundariesEta = cms.vdouble( -2.40, -2.08, -1.68, -1.26, -0.90, -0.62, -0.41, -0.20, 0.0, 0.20, 0.41, 0.62, 0.90, 1.26, 1.68, 2.08, 2.40 ) # defining r-z sector shape
    BoundariesEta = cms.vdouble( -2.50, -2.23, -1.88, -1.36, -0.90, -0.62, -0.41, -0.20, 0.0, 0.20, 0.41, 0.62, 0.90, 1.36, 1.88, 2.23, 2.50 ) # defining r-z sector shape
  ),

  # Parmeter specifying HoughTransform
  HoughTransform = cms.PSet (
    NumBinsInv2R = cms.int32( 16 ), # number of used inv2R bins
    NumBinsPhiT  = cms.int32( 32 ), # number of used phiT bins
    MinLayers    = cms.int32(  5 ), # required number of stub layers to form a candidate
    DepthMemory  = cms.int32( 32 )  # internal fifo depth
  ),

  # Parmeter specifying MiniHoughTransform
  MiniHoughTransform = cms.PSet (
    NumBinsInv2R  = cms.int32( 2 ), # number of finer inv2R bins inside HT bin
    NumBinsPhiT   = cms.int32( 2 ), # number of finer phiT bins inside HT bin
    NumDLBs       = cms.int32( 2 ), # number of dynamic load balancing steps
    NumDLBNodes   = cms.int32( 8 ), # number of units per dynamic load balancing step
    NumDLBChannel = cms.int32( 2 ), # number of inputs per dynamic load balancing unit
    MinLayers     = cms.int32( 5 )  # required number of stub layers to form a candidate
  ),

  # Parmeter specifying ZHoughTransform
  ZHoughTransform = cms.PSet (
    NumBinsZT        = cms.int32(  2 ), #
    NumBinsCot       = cms.int32(  2 ), #
    NumStages        = cms.int32(  5 ), #
    MinLayers        = cms.int32(  4 ), # required number of stub layers to form a candidate
    MaxTracks        = cms.int32( 16 ), # max number of output tracks per node
    MaxStubsPerLayer = cms.int32(  4 )  # cut on number of stub per layer for input candidates
  ),

  # Parmeter specifying KalmanFilter Input Formatter

  KalmanFilterIn = cms.PSet (
    ShiftRangePhi = cms.int32( 2 ), # power of 2 multiplier of stub phi residual range
    ShiftRangeZ   = cms.int32( 1 )  # power of 2 multiplier of stub z residual range
  ),

  # Parmeter specifying KalmanFilter
  KalmanFilter = cms.PSet (
    NumWorker   = cms.int32 ( 2   ), # number of kf worker
    RangeFactor = cms.double( 2.0 ), # search window of each track parameter in initial uncertainties
    MinLayers   = cms.int32 ( 4   ), # required number of stub layers to form a track
    MaxLayers   = cms.int32 ( 7   )  # maximum number of  layers added to a track
  ),

  # Parmeter specifying KalmanFilter Output Formatter
  KalmanFilterOut = cms.PSet (
    chi2rphiBins = cms.vdouble( 0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000,6000 ), # Final Chi2rphi digitization TODO extract from TTTrack Word 
    chi2rzBins   = cms.vdouble( 0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000,6000 ), # Final Chi2rz digitization TODO extract from TTTrack Word 

    chi2rphiConv = cms.int32 ( 3 ), # Conversion factor between dphi^2/weight and chi2rphi
    chi2rzConv   = cms.int32 ( 13 ), # Conversion factor between dz^2/weight and chi2rz

    WeightBinFraction = cms.int32( 0 ), # Number of bits dropped from dphi and dz for v0 and v1 LUTs

    TTTrackBits  = cms.int32( 96 )  # Number of bits for the tttrack word TODO extract from TTTrack_word dataformat
  ),

  # Parmeter specifying DuplicateRemoval
  DuplicateRemoval = cms.PSet (
    DepthMemory  = cms.int32( 16 ) # internal memory depth
  )

)
