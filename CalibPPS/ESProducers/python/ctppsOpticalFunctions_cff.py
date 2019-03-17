import FWCore.ParameterSet.Config as cms

from CalibPPS.ESProducers.ctppsLHCInfo_cff import *

from CalibPPS.ESProducers.ctppsOpticalFunctionsESSource_cfi import ctppsOpticalFunctionsESSource as _optics_tmp

# optical functions for 2016 data
ctppsOpticalFunctionsESSource_2016 = _optics_tmp.clone(
  validityRange = cms.EventRange("270293:min - 290872:max"),

  opticalFunctions = cms.VPSet(
      cms.PSet( xangle = cms.double(185), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions_2016.root") )
  ),

  scoringPlanes = cms.VPSet(
      # z in cm
      cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-20382.6) ),  # RP 002
      cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003
      cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+20382.6) ),  # RP 102
      cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103
  )
)

# optics interpolation between crossing angles
from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cfi import *
ctppsInterpolatedOpticalFunctionsESSource.lhcInfoLabel = ctppsLHCInfoLabel
