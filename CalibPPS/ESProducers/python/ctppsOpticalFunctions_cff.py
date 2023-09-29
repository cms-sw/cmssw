import FWCore.ParameterSet.Config as cms


# by default, (raw) optical functions are now loaded from CondDB using a GT

#from CalibPPS.ESProducers.ctppsOpticalFunctionsESSource_cfi import *
#
## add 2016 pre-TS2 configuration
#config_2016_preTS2 = cms.PSet(
#  validityRange = cms.EventRange("273725:min - 280385:max"),
#
#  opticalFunctions = cms.VPSet(
#      cms.PSet( xangle = cms.double(185), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions_2016.root") )
#  ),
#
#  scoringPlanes = cms.VPSet(
#      # z in cm
#      cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-20382.6) ),  # RP 002, strip
#      cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, strip
#      cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+20382.6) ),  # RP 102, strip
#      cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, strip
#  )
#)
#
#ctppsOpticalFunctionsESSource.configuration.append(config_2016_preTS2)

# optics interpolation between crossing angles
from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cff import *
