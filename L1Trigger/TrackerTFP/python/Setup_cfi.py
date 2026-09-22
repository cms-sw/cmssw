# configuration for TrackerTFPSetup

import FWCore.ParameterSet.Config as cms

TrackerTFP_params = cms.PSet (

  # Parmeter specifying TFP 
  TFP = cms.PSet (
    WidthPhi0  = cms.int32( 12 ), # number of bist used for phi0
    WidthInvR  = cms.int32( 15 ), # number of bist used for invR
    WidthCot   = cms.int32( 16 ), # number of bist used for cot(theta)
    WidthZ0    = cms.int32( 12 ), # number of bist used for z0
    NumChannel = cms.int32(  2 )  # number of output links
  ),

  # Parmeter specifying GeometricProcessor
  GeometricProcessor = cms.PSet (
    NumBinsPhiT = cms.int32 (   2    ), # number of phi sectors used in hough transform
    NumBinsZT   = cms.int32 (  32    ), # number of eta sectors used in hough transform
    ChosenRofZ  = cms.double(  57.76 ), # critical radius defining r-z sector shape in cm
    DepthMemory = cms.int32 (  32    ), # fifo depth in stub router firmware
  ),

  # Parmeter specifying HoughTransform
  HoughTransform = cms.PSet (
    NumBinsInv2R = cms.int32( 16 ), # number of used inv2R bins
    NumBinsPhiT  = cms.int32( 32 ), # number of used phiT bins
    MinLayers    = cms.int32(  5 ), # required number of stub layers to form a candidate
    DepthMemory  = cms.int32( 32 )  # internal fifo depth
  ),

  # Parmeter specifying Clean Track Builder

  CleanTrackBuilder = cms.PSet (
    NumBinsInv2R = cms.int32(  4 ), # number of inv2R bins
    NumBinsPhiT  = cms.int32(  4 ), # number of phiT bins
    NumBinsCot   = cms.int32(  4 ), # number of cot bins
    NumBinsZT    = cms.int32(  4 ), # number of zT bins
    MinLayers    = cms.int32(  4 ), # required number of stub layers to form a candidate
    MaxTracks    = cms.int32( 16 ), # max number of output tracks per node
    MaxStubs     = cms.int32(  4 ), # cut on number of stub per layer for input candidates
    DepthMemory  = cms.int32( 16 )  # internal fifo depth
  ),

  # Parmeter specifying KalmanFilter
  KalmanFilter = cms.PSet (
    NumWorker                = cms.int32 (   4   ), # number of kf worker
    MaxTracks                = cms.int32 (  63   ), # max number of tracks a kf worker can process
    MinLayers                = cms.int32 (   4   ), # required number of stub layers to form a track
    MaxLayers                = cms.int32 (   8   ), # maximum number of  layers added to a track
    MaxGaps                  = cms.int32 (   4   ), # maximum number of layer gaps allowed during cominatorical track building
    MaxSeedingLayer          = cms.int32 (   4   ), # perform seeding in layers 0 to this
    NumSeedStubs             = cms.int32 (   2   ), # number of stubs forming a seed
    ShiftChi20               = cms.int32 (  -1   ), # shiting chi2 in r-phi plane by power of two when caliclating chi2
    ShiftChi21               = cms.int32 (  -5   ), # shiting chi2 in r-z plane by power of two when caliclating chi2
    CutChi2                  = cms.double(   2.0 ), # cut on chi2 over degree of freedom
    WidthChi2                = cms.int32 (   8   )  # number of bits used to represent chi2 over degree of freedom
  ),

  # Parmeter specifying DuplicateRemoval
  DuplicateRemoval = cms.PSet (
    DepthMemory  = cms.int32( 16 ) # internal memory depth
  ),

  # Parameter specifying Track Quality
  TrackQuality = cms.PSet (
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

)
