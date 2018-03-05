import FWCore.ParameterSet.Config as cms

from CalibrTracker.SiPixelQuality.SiPixelStatusHarvester_cfi import *

# configure the PoolDBOutput service
from CondCore.DBCommon.CondDBCommon_cfi import *

CondDBCommon.connect = "sqlite_file:PromptCalibProdSiPixelConditions.db"
PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
                                  toPut = cms.VPSet(cms.PSet(
                                      record = cms.string('SiPixelQualityFromDbRcd'),
                                      tag = cms.string('TestLSBased') )),
                                  loadBlobStreamer = cms.untracked.bool(False),
                                  timetype   = cms.untracked.string('lumiid')
                                  )

