import FWCore.ParameterSet.Config as cms

# import the needed ingredients
from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripQualityHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsAAGHarvester_cff import *
from Alignment.CommonAlignmentProducer.AlcaSiPixelAliHarvester_cff import *
from Calibration.EcalCalibAlgos.AlcaEcalPedestalsHarvester_cff import *
from Calibration.LumiAlCaRecoProducers.AlcaLumiPCCHarvester_cff import *
from CalibTracker.SiPixelQuality.SiPixelStatusHarvester_cfi import *

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
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# BeamSpot by Run
ALCAHARVESTBeamSpotByRun = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("runbased")
ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByRun")

ALCAHARVESTBeamSpotByRun_metadata = cms.PSet(record = cms.untracked.string('BeamSpotObjectsRcdByRun'))

ALCAHARVESTBeamSpotByRun_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByRun'),
                                             tag = cms.string('BeamSpotObject_ByRun'),
                                             timetype   = cms.untracked.string('runnumber')
                                             )

# --------------------------------------------------------------------------------------
# BeamSpot by Lumi
ALCAHARVESTBeamSpotByLumi = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased")
ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdByLumi")

# configuration of DropBox metadata and DB output
ALCAHARVESTBeamSpotByLumi_metadata = cms.PSet(record = cms.untracked.string('BeamSpotObjectsRcdByLumi'))

ALCAHARVESTBeamSpotByLumi_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdByLumi'),
                                              tag = cms.string('BeamSpotObject_ByLumi'),
                                              timetype   = cms.untracked.string('lumiid')
                                              )

# --------------------------------------------------------------------------------------
# BeamSpot HP by Run
ALCAHARVESTBeamSpotHPByRun = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotHPByRun.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("runbased")
ALCAHARVESTBeamSpotHPByRun.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdHPByRun")
ALCAHARVESTBeamSpotHPByRun.AlcaBeamSpotHarvesterParameters.BeamSpotModuleName = cms.untracked.string('alcaBeamSpotProducerHP')

ALCAHARVESTBeamSpotHPByRun_metadata = cms.PSet(record = cms.untracked.string('BeamSpotObjectsRcdHPByRun'))

ALCAHARVESTBeamSpotHPByRun_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdHPByRun'),
                                             tag = cms.string('BeamSpotObjectHP_ByRun'),
                                             timetype   = cms.untracked.string('runnumber')
                                             )

# --------------------------------------------------------------------------------------
# BeamSpot HP by Lumi
ALCAHARVESTBeamSpotHPByLumi = alcaBeamSpotHarvester.clone()
ALCAHARVESTBeamSpotHPByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased")
ALCAHARVESTBeamSpotHPByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName = cms.untracked.string("BeamSpotObjectsRcdHPByLumi")
ALCAHARVESTBeamSpotHPByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotModuleName = cms.untracked.string('alcaBeamSpotProducerHP')
ALCAHARVESTBeamSpotHPByLumi.AlcaBeamSpotHarvesterParameters.DumpTxt = cms.untracked.bool(True)

# configuration of DropBox metadata and DB output
ALCAHARVESTBeamSpotHPByLumi_metadata = cms.PSet(record = cms.untracked.string('BeamSpotObjectsRcdHPByLumi'))

ALCAHARVESTBeamSpotHPByLumi_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdHPByLumi'),
                                              tag = cms.string('BeamSpotObjectHP_ByLumi'),
                                              timetype   = cms.untracked.string('lumiid')
                                              )

# --------------------------------------------------------------------------------------
# SiStrip Quality
ALCAHARVESTSiStripQuality_metadata = cms.PSet(record = cms.untracked.string('SiStripBadStripRcd'))

ALCAHARVESTSiStripQuality_dbOutput = cms.PSet(record = cms.string('SiStripBadStripRcd'),
                                              tag = cms.string('SiStripBadStrip_pcl'),
                                              timetype   = cms.untracked.string('runnumber')
                                              )

# --------------------------------------------------------------------------------------
# SiStrip Gains
ALCAHARVESTSiStripGains_metadata = cms.PSet(record = cms.untracked.string('SiStripApvGainRcd'))

ALCAHARVESTSiStripGains_dbOutput = cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                            tag = cms.string('SiStripApvGain_pcl'),
                                            timetype   = cms.untracked.string('runnumber')
                                            )

# --------------------------------------------------------------------------------------
# SiStrip Gains (AAG)
ALCAHARVESTSiStripGainsAAG_metadata = cms.PSet(record = cms.untracked.string('SiStripApvGainRcdAAG'))

ALCAHARVESTSiStripGainsAAG_dbOutput = cms.PSet(record = cms.string('SiStripApvGainRcdAAG'),
                                                         tag = cms.string('SiStripApvGainAAG_pcl'),
                                                         timetype   = cms.untracked.string('runnumber')
                                                         )

# --------------------------------------------------------------------------------------
# SiPixel Alignment
ALCAHARVESTSiPixelAli_metadata = cms.PSet(record = cms.untracked.string('TrackerAlignmentRcd'))

ALCAHARVESTSiPixelAli_dbOutput = cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                          tag = cms.string('SiPixelAli_pcl'),
                                          timetype   = cms.untracked.string('runnumber')
                                          )

# --------------------------------------------------------------------------------------
# ECAL Pedestals
ALCAHARVESTEcalPedestals_metadata = cms.PSet(record = cms.untracked.string('EcalPedestalsRcd'))

ALCAHARVESTEcalPedestals_dbOutput = cms.PSet(record = cms.string('EcalPedestalsRcd'),
                                             tag = cms.string('EcalPedestals_pcl'),
                                             timetype   = cms.untracked.string('runnumber')
                                             )

# --------------------------------------------------------------------------------------
# Lumi PCC
ALCAHARVESTLumiPCC_metadata = cms.PSet(record = cms.untracked.string('LumiCorrectionsRcd'))

ALCAHARVESTLumiPCC_dbOutput = cms.PSet(record = cms.string('LumiCorrectionsRcd'),
                                             tag = cms.string('LumiPCCCorrections_pcl'),
                                             timetype   = cms.untracked.string('lumiid')
                                             )



# SiPixel Quality
ALCAHARVESTSiPixelQuality = siPixelStatusHarvester.clone()
ALCAHARVESTSiPixelQuality.SiPixelStatusManagerParameters.outputBase = cms.untracked.string("dynamicLumibased")
ALCAHARVESTSiPixelQuality.SiPixelStatusManagerParameters.aveDigiOcc = cms.untracked.int32(20000)
ALCAHARVESTSiPixelQuality.debug = cms.untracked.bool(False)

ALCAHARVESTSiPixelQuality_metadata = cms.VPSet(cms.PSet(record = cms.untracked.string('SiPixelQualityFromDbRcd_prompt')),
                                               cms.PSet(record = cms.untracked.string('SiPixelQualityFromDbRcd_stuckTBM')),
                                               cms.PSet(record = cms.untracked.string('SiPixelQualityFromDbRcd_other')))
ALCAHARVESTSiPixelQuality_dbOutput = cms.VPSet(cms.PSet(record = cms.string('SiPixelQualityFromDbRcd_prompt'),
                                                        tag = cms.string('SiPixelQualityFromDbRcd_prompt'),
                                                        timetype = cms.untracked.string('lumiid')
                                                        ),
                                               cms.PSet(record = cms.string('SiPixelQualityFromDbRcd_stuckTBM'),
                                                        tag = cms.string('SiPixelQualityFromDbRcd_stuckTBM'),
                                                        timetype = cms.untracked.string('lumiid'),
                                                        ),
                                               cms.PSet(record = cms.string('SiPixelQualityFromDbRcd_other'),
                                                        tag = cms.string('SiPixelQualityFromDbRcd_other'),
                                                        timetype = cms.untracked.string('lumiid')
                                                        )
                                               )

if ALCAHARVESTSiPixelQuality.debug == cms.untracked.bool(True) :
   ALCAHARVESTSiPixelQuality_dbOutput.append(
       cms.PSet(record = cms.string('SiPixelQualityFromDbRcd_PCL'),
                tag = cms.string('SiPixelQualityFromDbRcd_PCL'),
                timetype = cms.untracked.string('lumiid')
                )
   )

# define all the paths
BeamSpotByRun  = cms.Path(ALCAHARVESTBeamSpotByRun)
BeamSpotByLumi = cms.Path(ALCAHARVESTBeamSpotByLumi)
BeamSpotHPByRun  = cms.Path(ALCAHARVESTBeamSpotHPByRun)
BeamSpotHPByLumi = cms.Path(ALCAHARVESTBeamSpotHPByLumi)
SiStripQuality = cms.Path(ALCAHARVESTSiStripQuality)
SiStripGains   = cms.Path(ALCAHARVESTSiStripGains)
SiPixelAli     = cms.Path(ALCAHARVESTSiPixelAli)
EcalPedestals  = cms.Path(ALCAHARVESTEcalPedestals)
SiStripGainsAAG = cms.Path(ALCAHARVESTSiStripGainsAAG)
LumiPCC = cms.Path(ALCAHARVESTLumiPCC)
SiPixelQuality = cms.Path(ALCAHARVESTSiPixelQuality)

ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(dqmSaver+pclMetadataWriter)

#promptCalibHarvest = cms.Path(alcaBeamSpotHarvester)
