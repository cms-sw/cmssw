# configuration of TrackTriggerTrackQuality

import FWCore.ParameterSet.Config as cms

TrackQuality_params = cms.PSet(
  # This emulation GBDT is optimised for the HYBRID_NEWKF emulation and works with the emulation of the KF out module
  # It is compatible with the HYBRID simulation and will give equivilant performance with this workflow
  Model = cms.FileInPath( "L1Trigger/TrackTrigger/data/L1_TrackQuality_GBDT_emulation_digitized.json" ),
  #Vector of strings of training features, in the order that the model was trained with
  FeatureNames = cms.vstring( ["tanl",
                               "z0_scaled",
                               "bendchi2_bin",
                               "nstub",
                               "nlaymiss_interior",
                               "chi2rphi_bin",
                               "chi2rz_bin"
                              ] ),
  BaseShiftCot      = cms.int32(     -7 ), #
  BaseShiftZ0       = cms.int32(     -6 ), #
  BaseShiftAPfixed  = cms.int32(     -5 ), #
  Chi2rphiConv      = cms.int32(      3 ), # Conversion factor between dphi^2/weight and chi2rphi
  Chi2rzConv        = cms.int32(     13 ), # Conversion factor between dz^2/weight and chi2rz
  WeightBinFraction = cms.int32(      0 ), # Number of bits dropped from dphi and dz for v0 and v1 LUTs
  DzTruncation      = cms.int32( 262144 ), # Constant used in FW to prevent 32-bit int overflow
  DphiTruncation    = cms.int32(     16 ), # Constant used in FW to prevent 32-bit int overflow
  WidthM20          = cms.int32(     16 ), #
  WidthM21          = cms.int32(     16 ), #
  WidthInvV0        = cms.int32(     16 ), #
  WidthInvV1        = cms.int32(     16 ), #
  Widthchi2rphi     = cms.int32(     20 ), #
  Widthchi2rz       = cms.int32(     20 ), #
  BaseShiftchi2rphi = cms.int32(     -6 ), #
  BaseShiftchi2rz   = cms.int32(     -6 )  #

)
