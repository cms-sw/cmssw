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
from Validation.CTPPS.year_2016.ctppsLHCInfoESSource_cfi import *

# TODO: load these data from DB
from Validation.CTPPS.year_2016.ctppsOpticalFunctionsESSource_cfi import *

recoCTPPSdets = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    totemTimingLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer *
    ctppsProtons
)
