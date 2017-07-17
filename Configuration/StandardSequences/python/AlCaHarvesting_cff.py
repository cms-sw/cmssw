import FWCore.ParameterSet.Config as cms

# import the needed ingredients
from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripQualityHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsAAGHarvester_cff import *
from Alignment.CommonAlignmentProducer.AlcaSiPixelAliHarvester_cff import *
from Calibration.EcalCalibAlgos.AlcaEcalPedestalsHarvester_cff import *

from Calibration.TkAlCaRecoProducers.PCLMetadataWriter_cfi import *

# common ingredients
from CondCore.CondDB.CondDB_cfi import CondDB
CondDBOutput = CondDB.clone(connect = cms.string("sqlite_file:promptCalibConditions.db"))

PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBOutput,
                                  toPut = cms.VPSet(),
                                  #timetype = cms.untracked.string("runnumber"),
                                  #timetype = cms.untracked.string("lumiid"),
                                  )


from DQMServices.Components.DQMFileSaver_cfi import * # FIXME
dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/Express/PCLTest/ALCAPROMPT'
#dqmSaver.saveAtJobEnd = True

# workflow definitions

ALCAHARVESTBeamSpotByRun = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("runbased")
ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByRun")

ALCAHARVESTBeamSpotByRun_metadata = cms.PSet(record = cms.untracked.string('BeamSpotObjectsRcdByRun'))

ALCAHARVESTBeamSpotByRun_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByRun'),
                                             tag = cms.string('BeamSpotObject_ByRun'),
                                             timetype   = cms.untracked.string('runnumber')
                                             )

ALCAHARVESTBeamSpotByLumi = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased")
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByLumi")

# configuration of DropBox metadata and DB output
ALCAHARVESTBeamSpotByLumi_metadata = cms.PSet(record = cms.untracked.string('BeamSpotObjectsRcdByLumi'))

ALCAHARVESTBeamSpotByLumi_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByLumi'),
                                              tag = cms.string('BeamSpotObject_ByLumi'),
                                              timetype   = cms.untracked.string('lumiid')
                                              )

# SiStrip Quality
ALCAHARVESTSiStripQuality_metadata = cms.PSet(record = cms.untracked.string('SiStripBadStripRcd'))

ALCAHARVESTSiStripQuality_dbOutput = cms.PSet(record = cms.string('SiStripBadStripRcd'),
                                              tag = cms.string('SiStripBadStrip_pcl'),
                                              timetype   = cms.untracked.string('runnumber')
                                              )

# SiStrip Gains
ALCAHARVESTSiStripGains_metadata = cms.PSet(record = cms.untracked.string('SiStripApvGainRcd'))

ALCAHARVESTSiStripGains_dbOutput = cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                            tag = cms.string('SiStripApvGain_pcl'),
                                            timetype   = cms.untracked.string('runnumber')
                                            )

# SiStrip Gains (AAG)
ALCAHARVESTSiStripGainsAAG_metadata = cms.PSet(record = cms.untracked.string('SiStripApvGainRcdAAG'))

ALCAHARVESTSiStripGainsAAG_dbOutput = cms.PSet(record = cms.string('SiStripApvGainRcdAAG'),
                                                         tag = cms.string('SiStripApvGainAAG_pcl'),
                                                         timetype   = cms.untracked.string('runnumber')
                                                         )

# SiPixel Alignment
ALCAHARVESTSiPixelAli_metadata = cms.PSet(record = cms.untracked.string('TrackerAlignmentRcd'))

ALCAHARVESTSiPixelAli_dbOutput = cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                          tag = cms.string('SiPixelAli_pcl'),
                                          timetype   = cms.untracked.string('runnumber')
                                          )

ALCAHARVESTEcalPedestals_metadata = cms.PSet(record = cms.untracked.string('EcalPedestalsRcd'))

ALCAHARVESTEcalPedestals_dbOutput = cms.PSet(record = cms.string('EcalPedestalsRcd'),
                                             tag = cms.string('EcalPedestals_pcl'),
                                             timetype   = cms.untracked.string('runnumber')
                                             )


# define the paths
BeamSpotByRun  = cms.Path(ALCAHARVESTBeamSpotByRun)
BeamSpotByLumi = cms.Path(ALCAHARVESTBeamSpotByLumi)
SiStripQuality = cms.Path(ALCAHARVESTSiStripQuality)
SiStripGains   = cms.Path(ALCAHARVESTSiStripGains)
SiPixelAli     = cms.Path(ALCAHARVESTSiPixelAli)
EcalPedestals  = cms.Path(ALCAHARVESTEcalPedestals)
SiStripGainsAAG = cms.Path(ALCAHARVESTSiStripGainsAAG)

ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(dqmSaver+pclMetadataWriter)

#promptCalibHarvest = cms.Path(alcaBeamSpotHarvester)
