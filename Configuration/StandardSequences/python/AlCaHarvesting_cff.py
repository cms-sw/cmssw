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
ALCAHARVESTBeamSpotByRun.metadataOfflineDropBox.destDB = cms.untracked.string("oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT")
ALCAHARVESTBeamSpotByRun.metadataOfflineDropBox.tag    = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_offline")
ALCAHARVESTBeamSpotByRun.metadataOfflineDropBox.DuplicateTagPROMPT    = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_prompt")
ALCAHARVESTBeamSpotByRun.metadataOfflineDropBox.DuplicateTagHLT    = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_hlt")
ALCAHARVESTBeamSpotByRun.metadataOfflineDropBox.IOVCheck   = cms.untracked.string("All")

ALCAHARVESTBeamSpotByRun_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByRun'),
                                             tag = cms.string('BeamSpotObject_ByRun'),
                                             timetype   = cms.untracked.string('runnumber'))


ALCAHARVESTBeamSpotByLumi = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased")
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByLumi")
ALCAHARVESTBeamSpotByLumi.metadataOfflineDropBox.destDB = cms.untracked.string("oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT")
ALCAHARVESTBeamSpotByLumi.metadataOfflineDropBox.tag    = cms.untracked.string("BeamSpotObjects_PCL_byLumi_v0_prompt")
ALCAHARVESTBeamSpotByLumi.metadataOfflineDropBox.DuplicateTagPROMPT    = cms.untracked.string("")
ALCAHARVESTBeamSpotByLumi_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByLumi'),
                                             tag = cms.string('BeamSpotObject_ByLumi'),
                                             timetype   = cms.untracked.string('lumiid'))



BeamSpotByRun = cms.Path(ALCAHARVESTBeamSpotByRun)
BeamSpotByLumi = cms.Path(ALCAHARVESTBeamSpotByLumi)


#promptCalibHarvest = cms.Path(alcaBeamSpotHarvester)



