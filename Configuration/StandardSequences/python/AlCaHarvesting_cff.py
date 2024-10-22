import FWCore.ParameterSet.Config as cms

# import the needed ingredients
from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripQualityHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripGainsAAGHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripHitEfficiencyHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiStripLorentzAngleHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiPixelLorentzAngleHarvester_cff import *
from Calibration.TkAlCaRecoProducers.AlcaSiPixelLorentzAngleHarvesterMCS_cff import *
from Alignment.CommonAlignmentProducer.AlcaSiPixelAliHarvester_cff import *
from Alignment.CommonAlignmentProducer.AlcaSiPixelAliHarvesterHG_cff import *
from Alignment.CommonAlignmentProducer.AlcaSiPixelAliHarvesterHGCombined_cff import *
from Calibration.EcalCalibAlgos.AlcaEcalPedestalsHarvester_cff import *
from Calibration.LumiAlCaRecoProducers.AlcaLumiPCCHarvester_cff import *
from CalibTracker.SiPixelQuality.SiPixelStatusHarvester_cfi import *
from CalibTracker.SiPixelQuality.DQMEventInfoSiPixelQuality_cff import *
from CalibPPS.TimingCalibration.PPSTimingCalibrationHarvester_cff import *
from CalibPPS.TimingCalibration.ALCARECOPPSDiamondSampicTimingCalibHarvester_cff import *
from CalibPPS.AlignmentGlobal.PPSAlignmentHarvester_cff import *

from CommonTools.ConditionDBWriter.PCLMetadataWriter_cfi import *

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
# BeamSpot HP - Low PU - by Run
ALCAHARVESTBeamSpotHPLowPUByRun = ALCAHARVESTBeamSpotHPByRun.clone()
ALCAHARVESTBeamSpotHPLowPUByRun.AlcaBeamSpotHarvesterParameters.BeamSpotModuleName = cms.untracked.string('alcaBeamSpotProducerHPLowPU')

# configuration of DropBox metadata and DB output
ALCAHARVESTBeamSpotHPLowPUByRun_metadata = cms.PSet(record = cms.untracked.string('BeamSpotObjectsRcdHPByRun'))

ALCAHARVESTBeamSpotHPLowPUByRun_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdHPByRun'),
                                                    tag = cms.string('BeamSpotObjectHP_ByRun'),
                                                    timetype   = cms.untracked.string('runnumber')
                                                    )

# --------------------------------------------------------------------------------------
# BeamSpot HP - Low PU - by Lumi
ALCAHARVESTBeamSpotHPLowPUByLumi = ALCAHARVESTBeamSpotHPByLumi.clone()
ALCAHARVESTBeamSpotHPLowPUByLumi.AlcaBeamSpotHarvesterParameters.BeamSpotModuleName = cms.untracked.string('alcaBeamSpotProducerHPLowPU')


# configuration of DropBox metadata and DB output
ALCAHARVESTBeamSpotHPLowPUByLumi_metadata = cms.PSet(record = cms.untracked.string('BeamSpotObjectsRcdHPByLumi'))

ALCAHARVESTBeamSpotHPLowPUByLumi_dbOutput = cms.PSet(record = cms.string('BeamSpotObjectsRcdHPByLumi'),
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
# SiStrip Lorentz Angle
ALCAHARVESTSiStripLA_metadata = cms.PSet(record = cms.untracked.string('SiStripLorentzAngleRcd'))

ALCAHARVESTSiStripLA_dbOutput = cms.PSet(record = cms.string('SiStripLorentzAngleRcd'),
                                         tag = cms.string('SiStripLA_pcl'),
                                         timetype   = cms.untracked.string('runnumber')
                                         )

# --------------------------------------------------------------------------------------
# SiStrip Bad Components from Hit Efficiency analysis
ALCAHARVESTSiStripHitEff_metadata = cms.PSet(record = cms.untracked.string('SiStripBadStripFromHitEffRcd'))

ALCAHARVESTSiStripHitEff_dbOutput = cms.PSet(record = cms.string('SiStripBadStripFromHitEffRcd'),
                                                    tag = cms.string('SiStripBadStripRcdHitEff_pcl'),
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
# SiPixel Alignment HG
ALCAHARVESTSiPixelAliHG_metadata = cms.PSet(record = cms.untracked.string('TrackerAlignmentHGRcd'))

ALCAHARVESTSiPixelAliHG_dbOutput = cms.PSet(record = cms.string('TrackerAlignmentHGRcd'),
                                          tag = cms.string('SiPixelAliHG_pcl'),
                                          timetype   = cms.untracked.string('runnumber')
                                          )

# --------------------------------------------------------------------------------------
# SiPixel Alignment HG combined
ALCAHARVESTSiPixelAliHGCombined_metadata = cms.PSet(record = cms.untracked.string('TrackerAlignmentHGCombinedRcd'))

ALCAHARVESTSiPixelAliHGCombined_dbOutput = cms.PSet(record = cms.string('TrackerAlignmentHGCombinedRcd'),
                                                    tag = cms.string('SiPixelAliHGCombined_pcl'),
                                                    timetype   = cms.untracked.string('runnumber')
                                                    )

# --------------------------------------------------------------------------------------
# SiPixel Lorentz Angle
ALCAHARVESTSiPixelLA_metadata = cms.PSet(record = cms.untracked.string('SiPixelLorentzAngleRcd'))

ALCAHARVESTSiPixelLA_dbOutput = cms.PSet(record = cms.string('SiPixelLorentzAngleRcd'),
                                         tag = cms.string('SiPixelLA_pcl'),
                                         timetype   = cms.untracked.string('runnumber')
                                         )

# --------------------------------------------------------------------------------------
# SiPixel Lorentz Angle Minimal Cluster Size
ALCAHARVESTSiPixelLAMCS_metadata = cms.PSet(record = cms.untracked.string('SiPixelLorentzAngleRcdMCS'))

ALCAHARVESTSiPixelLAMCS_dbOutput = cms.PSet(record = cms.string('SiPixelLorentzAngleRcdMCS'),
                                            tag = cms.string('SiPixelLAMCS_pcl'),
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
   dbOutput_ext = cms.VPSet(
        cms.PSet(record = cms.string('SiPixelQualityFromDbRcd_PCL'),
                tag = cms.string('SiPixelQualityFromDbRcd_PCL'),
                timetype = cms.untracked.string('lumiid')
                ),
        cms.PSet(record = cms.string('SiPixelQualityFromDbRcd_FEDerror25'),
                tag = cms.string('SiPixelQualityFromDbRcd_FEDerror25'),
                timetype = cms.untracked.string('lumiid'),
        ),
        cms.PSet(record = cms.string('SiPixelQualityFromDbRcd_permanentBad'),
                tag = cms.string('SiPixelQualityFromDbRcd_permanentBad'),
                timetype = cms.untracked.string('runnumber')
        )
   )
   ALCAHARVESTSiPixelQuality_dbOutput.extend(dbOutput_ext)

# --------------------------------------------------------------------------------------
# PPS calibration
ALCAHARVESTPPSTimingCalibration_metadata = cms.PSet(record = cms.untracked.string('PPSTimingCalibrationRcd_HPTDC'))
ALCAHARVESTPPSTimingCalibration_dbOutput = cms.PSet(record = cms.string('PPSTimingCalibrationRcd_HPTDC'),
                                                    tag = cms.string('PPSDiamondTimingCalibration_pcl'),
                                                    timetype = cms.untracked.string('runnumber')
                                                    )

ALCAHARVESTPPSDiamondSampicTimingCalibration_metadata = cms.PSet(record = cms.untracked.string('PPSTimingCalibrationRcd_SAMPIC'))
ALCAHARVESTPPSDiamondSampicTimingCalibration_dbOutput = cms.PSet(record = cms.string('PPSTimingCalibrationRcd_SAMPIC'),
                                            tag = cms.string('PPSDiamondSampicCalibration_pcl'),
                                            timetype = cms.untracked.string('runnumber'))

ALCAHARVESTPPSAlignment_metadata = cms.PSet(record = cms.untracked.string('CTPPSRPAlignmentCorrectionsDataRcd'))
ALCAHARVESTPPSAlignment_dbOutput = cms.PSet(record = cms.string('CTPPSRPAlignmentCorrectionsDataRcd'),
                                            tag = cms.string('CTPPSRPAlignment_real_pcl'),
                                            timetype = cms.untracked.string('runnumber'))

# define all the paths
BeamSpotByRun  = cms.Path(ALCAHARVESTBeamSpotByRun)
BeamSpotByLumi = cms.Path(ALCAHARVESTBeamSpotByLumi)
BeamSpotHPByRun  = cms.Path(ALCAHARVESTBeamSpotHPByRun)
BeamSpotHPByLumi = cms.Path(ALCAHARVESTBeamSpotHPByLumi)
BeamSpotHPLowPUByRun  = cms.Path(ALCAHARVESTBeamSpotHPLowPUByRun)
BeamSpotHPLowPUByLumi = cms.Path(ALCAHARVESTBeamSpotHPLowPUByLumi)
SiStripQuality = cms.Path(ALCAHARVESTSiStripQuality)
SiStripGains   = cms.Path(ALCAHARVESTSiStripGains)
SiStripGainsAAG = cms.Path(ALCAHARVESTSiStripGainsAAG)
SiStripHitEff = cms.Path(ALCAHARVESTSiStripHitEfficiency)
SiStripLA      = cms.Path(ALCAHARVESTSiStripLorentzAngle)
SiPixelAli     = cms.Path(ALCAHARVESTSiPixelAli)
SiPixelAliHG     = cms.Path(ALCAHARVESTSiPixelAliHG)
SiPixelAliHGCombined = cms.Path(ALCAHARVESTSiPixelAliHGCombined)
SiPixelLA      = cms.Path(ALCAHARVESTSiPixelLorentzAngle)
SiPixelLAMCS      = cms.Path(ALCAHARVESTSiPixelLorentzAngleMCS)
EcalPedestals  = cms.Path(ALCAHARVESTEcalPedestals)
LumiPCC = cms.Path(ALCAHARVESTLumiPCC)
SiPixelQuality = cms.Path(dqmEnvSiPixelQuality+ALCAHARVESTSiPixelQuality)#+siPixelPhase1DQMHarvester)
PPSTimingCalibration = cms.Path(ALCAHARVESTPPSTimingCalibration)
PPSDiamondSampicTimingCalibration = cms.Path(ALCAHARVESTPPSDiamondSampicTimingCalibration)
PPSAlignment = cms.Path(ALCAHARVESTPPSAlignment)

ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(dqmSaver+pclMetadataWriter)

#promptCalibHarvest = cms.Path(alcaBeamSpotHarvester)
