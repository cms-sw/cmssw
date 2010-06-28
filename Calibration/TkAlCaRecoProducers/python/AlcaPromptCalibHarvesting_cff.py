import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cfi import *

alcaBeamSpotHarvester.BeamSpotOutputBase = cms.untracked.string("lumibased") # runbased - lumibased

# configure the PoolDBOutput service

from CondCore.DBCommon.CondDBCommon_cfi import *

CondDBCommon.connect = "sqlite_file:PromptCalibConditions.db"
PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
                                  toPut = cms.VPSet(cms.PSet(
                                      record = cms.string('BeamSpotObjectsRcd'),
                                      tag = cms.string('TestLSBasedBS') )),
                                  loadBlobStreamer = cms.untracked.bool(False),
                                  timetype   = cms.untracked.string('lumiid')
                                  )

