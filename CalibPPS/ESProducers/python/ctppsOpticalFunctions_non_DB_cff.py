import FWCore.ParameterSet.Config as cms

# (source) optical functions sampled at few xangles
from CalibPPS.ESProducers.ctppsOpticalFunctionsESSource_cfi import *

optics_2016_preTS2 = cms.PSet(
  validityRange = cms.EventRange("273725:min - 280385:max"),

  opticalFunctions = cms.VPSet(
    cms.PSet( xangle = cms.double(185), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2016_preTS2/version2/185urad.root") )
  ),

  scoringPlanes = cms.VPSet(
    # z in cm
    cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-20382.7) ),  # RP 002, strip
    cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.0) ),  # RP 003, strip

    cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+20382.7) ),  # RP 102, strip
    cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.0) ),  # RP 103, strip
  )
)

ctppsOpticalFunctionsESSource.configuration.append(optics_2016_preTS2)

optics_2016_postTS2 = cms.PSet(
  validityRange = cms.EventRange("282730:min - 284044:max"),

  opticalFunctions = cms.VPSet(
    cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2016_postTS2/version2/140urad.root") )
  ),

  scoringPlanes = cms.VPSet(
    # z in cm
    cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-20382.7) ),  # RP 002, strip
    cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.0) ),  # RP 003, strip

    cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+20382.7) ),  # RP 102, strip
    cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.0) ),  # RP 103, strip
  )
)

ctppsOpticalFunctionsESSource.configuration.append(optics_2016_postTS2)

optics_2017 = cms.PSet(
  validityRange = cms.EventRange("297046:min - 307082:max"),

  opticalFunctions = cms.VPSet(
    cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/120urad.root") ),
    cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/130urad.root") ),
    cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/140urad.root") )
  ),

  scoringPlanes = cms.VPSet(
    # z in cm
    cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.0) ),  # RP 003, strip
    cms.PSet( rpId = cms.uint32(2054160384), dirName = cms.string("XRPH_E6L5_B2"), z = cms.double(-21570.0) ),  # RP 016, diamond
    cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel

    cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.0) ),  # RP 103, strip
    cms.PSet( rpId = cms.uint32(2070937600), dirName = cms.string("XRPH_E6R5_B1"), z = cms.double(+21570.0) ),  # RP 116, diamond
    cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
  )
)

ctppsOpticalFunctionsESSource.configuration.append(optics_2017)

optics_2018 = cms.PSet(
  validityRange = cms.EventRange("314747:min - 325175:max"),

  opticalFunctions = cms.VPSet(
    cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version6/120urad.root") ),
    cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version6/130urad.root") ),
    cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version6/140urad.root") )
  ),

  scoringPlanes = cms.VPSet(
    # z in cm
    cms.PSet( rpId = cms.uint32(2014838784), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.0) ),  # RP 003, pixel
    cms.PSet( rpId = cms.uint32(2054160384), dirName = cms.string("XRPH_E6L5_B2"), z = cms.double(-21570.0) ),  # RP 016, diamond
    cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel

    cms.PSet( rpId = cms.uint32(2031616000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.0) ),  # RP 103, pixel
    cms.PSet( rpId = cms.uint32(2070937600), dirName = cms.string("XRPH_E6R5_B1"), z = cms.double(+21570.0) ),  # RP 116, diamond
    cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
  )
)

ctppsOpticalFunctionsESSource.configuration.append(optics_2018)

optics_2021 = cms.PSet(
  validityRange = cms.EventRange("1234:1 - 1234:max"), # NB: a fake IOV, this optics was never used for LHC

  opticalFunctions = cms.VPSet(
    cms.PSet( xangle = cms.double(110.444), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2021/version_pre3/110.444urad.root") ),
    cms.PSet( xangle = cms.double(184.017), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2021/version_pre3/184.017urad.root") )
  ),

  scoringPlanes = cms.VPSet(
    # z in cm
    cms.PSet( rpId = cms.uint32(2014838784), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.0) ),  # RP 003, pixel
    cms.PSet( rpId = cms.uint32(2056257536), dirName = cms.string("XRPH_A6L5_B2"), z = cms.double(-21507.8) ),  # RP 022, diamond
    cms.PSet( rpId = cms.uint32(2054160384), dirName = cms.string("XRPH_E6L5_B2"), z = cms.double(-21570.0) ),  # RP 016, diamond
    cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel

    cms.PSet( rpId = cms.uint32(2031616000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.0) ),  # RP 103, pixel
    cms.PSet( rpId = cms.uint32(2073034752), dirName = cms.string("XRPH_A6R5_B1"), z = cms.double(+21507.8) ),  # RP 122, diamond
    cms.PSet( rpId = cms.uint32(2070937600), dirName = cms.string("XRPH_E6R5_B1"), z = cms.double(+21570.0) ),  # RP 116, diamond
    cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
  )
)

# NB: do not append the 2021 config - not used for any LHC data

optics_2022 = cms.PSet(
  validityRange = cms.EventRange("343890:min - 999999:max"),

  opticalFunctions = cms.VPSet(
    cms.PSet( xangle = cms.double(144.974), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2022/version_pre1/144.974urad.root") ),
    cms.PSet( xangle = cms.double(160.000), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2022/version_pre1/160.000urad.root") )
  ),

  scoringPlanes = cms.VPSet(
    # z in cm
    cms.PSet( rpId = cms.uint32(2014838784), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.0) ),  # RP 003, pixel
    cms.PSet( rpId = cms.uint32(2056257536), dirName = cms.string("XRPH_A6L5_B2"), z = cms.double(-21507.8) ),  # RP 022, diamond
    cms.PSet( rpId = cms.uint32(2054160384), dirName = cms.string("XRPH_E6L5_B2"), z = cms.double(-21570.0) ),  # RP 016, diamond
    cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel

    cms.PSet( rpId = cms.uint32(2031616000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.0) ),  # RP 103, pixel
    cms.PSet( rpId = cms.uint32(2073034752), dirName = cms.string("XRPH_A6R5_B1"), z = cms.double(+21507.8) ),  # RP 122, diamond
    cms.PSet( rpId = cms.uint32(2070937600), dirName = cms.string("XRPH_E6R5_B1"), z = cms.double(+21570.0) ),  # RP 116, diamond
    cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
  )
)

ctppsOpticalFunctionsESSource.configuration.append(optics_2022)

# optics interpolation between crossing angles
from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cff import *
