import FWCore.ParameterSet.Config as cms

# import the needed ingredients
from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripQualityHarvester_cff import *
from Calibration.TkAlCaRecoProducers.PCLMetadataWriter_cfi import *

# commoon ingredients
from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
CondDBCommon.connect = "sqlite_file:promptCalibConditions.db"

PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
                                  toPut = cms.VPSet(),
                                  loadBlobStreamer = cms.untracked.bool(False),
                                  #    timetype   = cms.untracked.string('lumiid')
                                  #    timetype   = cms.untracked.string('runnumber')
                                  )


from DQMServices.Components.DQMFileSaver_cfi import * # FIXME
dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/Express/PCLTest/ALCAPROMPT'
#dqmSaver.saveAtJobEnd = True

# workflow definitions

ALCAHARVESTBeamSpotByRun = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("runbased")
ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByRun")


ALCAHARVESTBeamSpotByRun_metadata = cms.PSet(record              = cms.untracked.string('BeamSpotObjectsRcdByRun'),
                                             destDB              = cms.untracked.string("oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT"),
                                             destDBValidation    = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT"),
                                             tag                 = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_offline"),
                                             Timetype            = cms.untracked.string("runnumber"),
                                             IOVCheck            = cms.untracked.string("All"),
                                             DuplicateTagHLT     = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_hlt"),
                                             DuplicateTagEXPRESS = cms.untracked.string(""),
                                             DuplicateTagPROMPT  = cms.untracked.string("BeamSpotObjects_PCL_byRun_v0_prompt"),
                                             )


ALCAHARVESTBeamSpotByRun_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByRun'),
                                             tag = cms.string('BeamSpotObject_ByRun'),
                                             timetype   = cms.untracked.string('runnumber'))


ALCAHARVESTBeamSpotByLumi = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased")
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByLumi")


ALCAHARVESTBeamSpotByLumi_metadata = cms.PSet(record              = cms.untracked.string('BeamSpotObjectsRcdByLumi'),
                                              destDB              = cms.untracked.string("oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT"),
                                              destDBValidation    = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT"),
                                              tag                 = cms.untracked.string("BeamSpotObjects_PCL_byLumi_v0_offline"),
                                              Timetype            = cms.untracked.string("lumiid"),
                                              IOVCheck            = cms.untracked.string("All"),
                                              DuplicateTagHLT     = cms.untracked.string("BeamSpotObjects_PCL_byLumi_v0_hlt"),
                                              DuplicateTagEXPRESS = cms.untracked.string(""),
                                              DuplicateTagPROMPT  = cms.untracked.string("BeamSpotObjects_PCL_byLumi_v0_prompt"),
                                              )

ALCAHARVESTBeamSpotByLumi_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByLumi'),
                                             tag = cms.string('BeamSpotObject_ByLumi'),
                                             timetype   = cms.untracked.string('lumiid'))


ALCAHARVESTSiStripQuality_metadata = cms.PSet(record              = cms.untracked.string('SiStripBadStripRcd'),
                                              destDB              = cms.untracked.string("oracle://cms_orcon_prod/CMS_COND_31X_STRIP"),
                                              destDBValidation    = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_STRIP"),
                                              tag                 = cms.untracked.string("SiStripBadChannel_PCL_v0_offline"),
                                              Timetype            = cms.untracked.string("runnumber"),
                                              IOVCheck            = cms.untracked.string("All"),
                                              DuplicateTagHLT     = cms.untracked.string("SiStripBadChannel_PCL_v0_hlt"),
                                              DuplicateTagEXPRESS = cms.untracked.string(""),
                                              DuplicateTagPROMPT  = cms.untracked.string("SiStripBadChannel_PCL_v0_prompt"),
                                              )


ALCAHARVESTSiStripQuality_dbOutput = cms.PSet(record = cms.string('SiStripBadStripRcd'),
                                             tag = cms.string('SiStripBadStrip_pcl'),
                                             timetype   = cms.untracked.string('runnumber'))


# define the paths

BeamSpotByRun = cms.Path(ALCAHARVESTBeamSpotByRun)
BeamSpotByLumi = cms.Path(ALCAHARVESTBeamSpotByLumi)
SiStripQuality = cms.Path(ALCAHARVESTSiStripQuality)

ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(dqmSaver+pclMetadataWriter)

#promptCalibHarvest = cms.Path(alcaBeamSpotHarvester)



