import FWCore.ParameterSet.Config as cms

# import the needed ingredients
from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripQualityHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsAfterAbortGapHarvester_cff import *
from Alignment.CommonAlignmentProducer.AlcaSiPixelAliHarvester_cff import *

from Calibration.TkAlCaRecoProducers.PCLMetadataWriter_cfi import *

# common ingredients
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
                                             )


ALCAHARVESTBeamSpotByRun_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByRun'),
                                             tag = cms.string('BeamSpotObject_ByRun'),
                                             timetype   = cms.untracked.string('runnumber'))


ALCAHARVESTBeamSpotByLumi = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased")
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByLumi")

# configuration of DropBox metadata and DB output
ALCAHARVESTBeamSpotByLumi_metadata = cms.PSet(record              = cms.untracked.string('BeamSpotObjectsRcdByLumi'),
                                              )

ALCAHARVESTBeamSpotByLumi_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByLumi'),
                                             tag = cms.string('BeamSpotObject_ByLumi'),
                                             timetype   = cms.untracked.string('lumiid'))


ALCAHARVESTSiStripQuality_metadata = cms.PSet(record              = cms.untracked.string('SiStripBadStripRcd'),
                                              )


ALCAHARVESTSiStripQuality_dbOutput = cms.PSet(record = cms.string('SiStripBadStripRcd'),
                                             tag = cms.string('SiStripBadStrip_pcl'),
                                             timetype   = cms.untracked.string('runnumber'))


ALCAHARVESTSiStripGains_metadata = cms.PSet(record              = cms.untracked.string('SiStripApvGainRcd'),
                                              )


ALCAHARVESTSiStripGains_dbOutput = cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                             tag = cms.string('SiStripApvGain_pcl'),
                                             timetype   = cms.untracked.string('runnumber'))

ALCAHARVESTSiStripGainsAfterAbortGap_metadata = cms.PSet(record = cms.untracked.string('SiStripApvGainRcd'),
                                                        )

ALCAHARVESTSiStripGainsAfterAbortGap_dbOutput = cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                            tag = cms.string('SiStripApvGainAfterAbortGap_pcl'),
                                                     timetype   = cms.untracked.string('runnumber'))

    #
ALCAHARVESTSiPixelAli_metadata = cms.PSet(record              = cms.untracked.string('TrackerAlignmentRcd'),
                                              )


ALCAHARVESTSiPixelAli_dbOutput = cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                          tag = cms.string('SiPixelAli_pcl'),
                                          timetype   = cms.untracked.string('runnumber'))




# define the paths

BeamSpotByRun  = cms.Path(ALCAHARVESTBeamSpotByRun)
BeamSpotByLumi = cms.Path(ALCAHARVESTBeamSpotByLumi)
SiStripQuality = cms.Path(ALCAHARVESTSiStripQuality)
SiStripGains   = cms.Path(ALCAHARVESTSiStripGains)
SiPixelAli     = cms.Path(ALCAHARVESTSiPixelAli)

SiStripGainsAfterAbortGap = cms.Path(ALCAHARVESTSiStripGainsAfterAbortGap)


ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(dqmSaver+pclMetadataWriter)

#promptCalibHarvest = cms.Path(alcaBeamSpotHarvester)



