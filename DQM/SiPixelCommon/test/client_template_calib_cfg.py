import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

process = cms.Process("SIPIXELDQM")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")


process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True

process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")

process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi")
process.SiPixelRawDataErrorSource.saveFile = False
process.SiPixelRawDataErrorSource.isPIB = False
process.SiPixelRawDataErrorSource.slowDown = False
process.SiPixelRawDataErrorSource.reducedSet = False

process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")
process.SiPixelDigiSource.saveFile = False
process.SiPixelDigiSource.isPIB = False
process.SiPixelDigiSource.slowDown = False
process.SiPixelDigiSource.modOn = True
process.SiPixelDigiSource.twoDimOn = True
process.SiPixelDigiSource.hiRes = True
process.SiPixelDigiSource.ladOn = False
process.SiPixelDigiSource.layOn = False
process.SiPixelDigiSource.phiOn = False
process.SiPixelDigiSource.bladeOn = False
process.SiPixelDigiSource.diskOn = False
process.SiPixelDigiSource.ringOn = False

process.load("DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi")
process.SiPixelClusterSource.saveFile = False
process.SiPixelClusterSource.isPIB = False
process.SiPixelClusterSource.modOn = True
process.SiPixelClusterSource.twoDimOn = True
process.SiPixelClusterSource.reducedSet = False
process.SiPixelClusterSource.ladOn = False
process.SiPixelClusterSource.layOn = False
process.SiPixelClusterSource.phiOn = False
process.SiPixelClusterSource.bladeOn = False
process.SiPixelClusterSource.diskOn = False
process.SiPixelClusterSource.ringOn = False

process.load("DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi")
process.SiPixelRecHitSource.saveFile = False
process.SiPixelRecHitSource.isPIB = False
process.SiPixelRecHitSource.modOn = True
process.SiPixelRecHitSource.twoDimOn = True
process.SiPixelRecHitSource.ladOn = False
process.SiPixelRecHitSource.layOn = False
process.SiPixelRecHitSource.phiOn = False
process.SiPixelRecHitSource.bladeOn = False
process.SiPixelRecHitSource.ringOn = False
process.SiPixelRecHitSource.diskOn = False

process.load("CalibTracker.SiPixelTools.SiPixelErrorsCalibDigis_cfi")
process.siPixelErrorsDigisToCalibDigis.saveFile=False
process.siPixelErrorsDigisToCalibDigis.SiPixelProducerLabelTag = 'siPixelCalibDigis'
process.load("CalibTracker.SiPixelGainCalibration.SiPixelCalibDigiProducer_cfi")
process.load("CalibTracker.SiPixelSCurveCalibration.SiPixelSCurveCalibrationAnalysis_cfi")
process.siPixelSCurveAnalysis.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelSCurveAnalysis.saveFile = False
process.load("CalibTracker.SiPixelIsAliveCalibration.SiPixelIsAliveCalibration_cfi")
process.siPixelIsAliveCalibration.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelIsAliveCalibration.saveFile = False
process.load("CalibTracker.SiPixelGainCalibration.SiPixelGainCalibrationAnalysis_cfi")
process.siPixelGainCalibrationAnalysis.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelGainCalibrationAnalysis.saveFile = False
process.siPixelGainCalibrationAnalysis.savePixelLevelHists = False

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"

from CondCore.DBCommon.CondDBCommon_cfi import *
process.siPixelCalibGlobalTag =  cms.ESSource("PoolDBESSource",
                                              CondDBCommon,
                                              connect = cms.string("frontier://FrontierProd/CMS_COND_21X_PIXEL"),
                                              
                                              toGet = cms.VPSet(
                                                cms.PSet(record = cms.string('SiPixelCalibConfigurationRcd'),
                                                tag = cms.string('GLOBALCALIB_default'))
                                                )
                                              )
process.es_prefer_dbcalib = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
                            ONEPARAM
                            TWOPARAM
    fileNames = cms.untracked.vstring(
    'FILENAME'
       				      )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    TEXTFILE = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('TEXTFILE')
)

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.sipixelEDAClient = DQMEDHarvester("SiPixelEDAClient",
    EventOffsetForInit = cms.untracked.int32(10),
    ActionOnLumiSection = cms.untracked.bool(False),
    ActionOnRunEnd = cms.untracked.bool(True),
    HighResolutionOccupancy = cms.untracked.bool(True),
    NoiseRateCutValue = cms.untracked.double(-1.)
    
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.Reco = cms.Sequence(process.siPixelDigis)
process.Calibration = cms.Sequence(process.siPixelCalibDigis*process.siPixelGainCalibrationAnalysis*process.siPixelIsAliveCalibration)
process.RAWmonitor = cms.Sequence(process.SiPixelRawDataErrorSource)
process.DIGImonitor = cms.Sequence(process.SiPixelDigiSource)
process.CLUmonitor = cms.Sequence(process.SiPixelClusterSource)
process.HITmonitor = cms.Sequence(process.SiPixelRecHitSource)
process.DQMmodules = cms.Sequence(process.qTester*process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.Reco*process.qTester*process.dqmEnv*process.Calibration*process.sipixelEDAClient*process.dqmSaver)

# cms.Path(process.Reco*process.DQMmodules*process.Calibration*process.RAWmonitor*process.DIGImonitprocess.sipixelEDAClient)
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Pixel'
process.dqmSaver.dirName = '.'
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
