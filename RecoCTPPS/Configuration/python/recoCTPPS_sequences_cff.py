import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.totemTimingLocalReconstruction_cff import *
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer

from RecoCTPPS.ProtonReconstruction.ctppsProtons_cfi import *

# TODO: get this from standard DB and GT
from CondCore.CondDB.CondDB_cfi import *
CondDB.connect = 'frontier://FrontierPrep/CMS_CONDITIONS'
PoolDBESSource = cms.ESSource("PoolDBESSource",
  CondDB,
  DumpStat = cms.untracked.bool(False),
  toGet = cms.VPSet(
    cms.PSet(
      record = cms.string("RPRealAlignmentRecord"),
      tag = cms.string("CTPPSRPAlignmentCorrections_real_test")
    )
  )
)

# TODO: load these data from DB
# NB: this example only works for 2016 data
ctppsLHCInfoESSource = cms.ESSource("CTPPSLHCInfoESSource",
  validityRange = cms.EventRange("270293:min - 290872:max"),
  beamEnergy = cms.double(6500),  # GeV
  xangle = cms.double(185)  # murad
)

# TODO: load these data from DB
# NB: this example only works for 2016 data
import FWCore.ParameterSet.Config as cms
from CalibPPS.ESProducers.ctppsOpticalFunctionsESSource_cfi import ctppsOpticalFunctionsESSource as _optics_tmp
ctppsOpticalFunctionsESSource = _optics_tmp.clone(
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

ctppsInterpolatedOpticalFunctionsESSource = cms.ESProducer("CTPPSInterpolatedOpticalFunctionsESSource")

recoCTPPSdets = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    totemTimingLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer *
    ctppsProtons
)
