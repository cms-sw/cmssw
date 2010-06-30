import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff import *

from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
CondDBCommon.connect = "sqlite_file:promptCalibConditions.db"

# FIXME: the toPut could be configured dinamically....
PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
                                  toPut = cms.VPSet(cms.PSet(record = cms.string('BeamSpotObjectsRcd'),
                                                             tag = cms.string('BeamSpotObject_runbased'),
                                                             timetype   = cms.untracked.string('runnumber'))
                                                    ),
                                  loadBlobStreamer = cms.untracked.bool(False),
                                  #    timetype   = cms.untracked.string('lumiid')
                                  #    timetype   = cms.untracked.string('runnumber')
                                  )

promptCalibHarvest = cms.Path(alcaBeamSpotHarvester)



