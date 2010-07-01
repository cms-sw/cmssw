import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff import *

from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
CondDBCommon.connect = "sqlite_file:promptCalibConditions.db"

PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
                                  toPut = cms.VPSet(),
                                  loadBlobStreamer = cms.untracked.bool(False),
                                  #    timetype   = cms.untracked.string('lumiid')
                                  #    timetype   = cms.untracked.string('runnumber')
                                  )


ALCAHARVESTBeamSpotByRun = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("runbased")
ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByRun")
ALCAHARVESTBeamSpotByRun_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByRun'),
                                             tag = cms.string('BeamSpotObject_ByRun'),
                                             timetype   = cms.untracked.string('runnumber'))


ALCAHARVESTBeamSpotByLumi = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased")
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByLumi")
ALCAHARVESTBeamSpotByLumi_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByLumi'),
                                             tag = cms.string('BeamSpotObject_ByLumi'),
                                             timetype   = cms.untracked.string('lumiid'))



BeamSpotByRun = cms.Path(ALCAHARVESTBeamSpotByRun)
BeamSpotByLumi = cms.Path(ALCAHARVESTBeamSpotByLumi)


#promptCalibHarvest = cms.Path(alcaBeamSpotHarvester)



