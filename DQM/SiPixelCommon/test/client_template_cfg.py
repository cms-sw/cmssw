import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

process = cms.Process("SIPIXELDQM")

# load all appropriate modules:
# get alignment conditions needed for geometry:
process.load("Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi")
process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")
process.load("DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi")
process.load("DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("CalibTracker.SiPixelTools.SiPixelErrorsCalibDigis_cfi")
process.load("CalibTracker.SiPixelGainCalibration.SiPixelCalibDigiProducer_cfi")
process.load("CalibTracker.SiPixelSCurveCalibration.SiPixelSCurveCalibrationAnalysis_cfi")
process.load("CalibTracker.SiPixelIsAliveCalibration.SiPixelIsAliveCalibration_cfi")
process.load("CalibTracker.SiPixelGainCalibration.SiPixelGainCalibrationAnalysis_cfi")

# get the global tag with all cabling maps, alignment info, etc.
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")

process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GTAG"


# and access the calibration information:
CALIBfrom CondCore.DBCommon.CondDBCommon_cfi import *
#                                              
CALIBprocess.siPixelCalibGlobalTag =  cms.ESSource("PoolDBESSource",
CALIB                                              CondDBCommon,
CALIB                                              connect = cms.string("oracle://cms_orcoff_prep/CMS_COND_PIXEL_COMM_21X"),
CALIB                                              globaltag = cms.string("PIXELCALIB_01::TypeGLOBALCALIB"),
CALIB                                              BlobStreamerName = cms.untracked.string("TBufferBlobStreamingService")
CALIB                                              )
CALIBprocess.siPixelCalibGlobalTag.DBParameters.authenticationPath = "/afs/cern.ch/cms/DB/conddb"

process.esprefer_dbcalib = cms.ESPrefer("PoolDBESSource","GlobalTag")

# this is needed by the gain calibration analyzer 
process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

# the input file source
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    ONEPARAM
    TWOPARAM
    fileNames = cms.untracked.vstring('FILENAME')
                            )


process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(-1)
    input = cms.untracked.int32(-1)
)

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('siPixelDigis', 
                                                                         'SiPixelRawDataErrorSource',
                                                                         'SiPixelCalibProducer',
                                                                         'SiPixelDigiSource', 
                                                                         'SiPixelClusterSource', 
                                                                         'SiPixelRecHitSource', 
                                                                         'sipixelEDAClient'),
                                    TEXTFILE = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')
                                                                           ),
                                    
#                                   destinations = cms.untracked.vstring('TEXTFILE')
                                    )

process.AdaptorConfig = cms.Service("AdaptorConfig")

# DQM modules:
process.sipixelEDAClient = DQMEDHarvester("SiPixelEDAClient",
    FileSaveFrequency = cms.untracked.int32(50),
    StaticUpdateFrequency = cms.untracked.int32(10)
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

# define all paths and sequences:
process.Digis = cms.Sequence(process.siPixelDigis)
process.Clusters = cms.Sequence(process.siPixelClusters)
process.Calibration = cms.Sequence(process.siPixelCalibDigis*process.siPixelErrorsDigisToCalibDigis*process.siPixelGainCalibrationAnalysis*process.siPixelIsAliveCalibration*process.siPixelSCurveAnalysis)
process.RAWmonitor = cms.Sequence(process.SiPixelRawDataErrorSource)
process.DIGImonitor = cms.Sequence(process.SiPixelDigiSource)
process.CLUmonitor = cms.Sequence(process.SiPixelClusterSource)
process.HITmonitor = cms.Sequence(process.SiPixelRecHitSource)
process.DQMmodules = cms.Sequence(process.qTester*process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.DQMmodules*DIGISPOTCLUSPOTRECSPOTCDSPOTSCURVESPOTGAINSPOTPIXELSPOTRAWMONSPOTDIGMONSPOTCLUMONSPOTRECMONSPOTprocess.sipixelEDAClient)
# choose one of these two:
# online-style DQM (runs RECO)

# offline-style DQM (reco in input file)
#process.p = cms.Path(process.DQMmodules*process.DIGImonitor*process.sipixelEDAClient)

# list of replace statements
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True
process.SiPixelRawDataErrorSource.saveFile = False
process.SiPixelRawDataErrorSource.isPIB = False
process.SiPixelRawDataErrorSource.slowDown = False
process.SiPixelDigiSource.saveFile = False
process.SiPixelDigiSource.isPIB = False
process.SiPixelDigiSource.slowDown = False
process.SiPixelDigiSource.modOn = True
process.SiPixelDigiSource.ladOn = False
process.SiPixelDigiSource.layOn = False
process.SiPixelDigiSource.phiOn = False
process.SiPixelDigiSource.bladeOn = False
process.SiPixelDigiSource.diskOn = False
process.SiPixelDigiSource.ringOn = False
process.SiPixelClusterSource.saveFile = False
process.SiPixelClusterSource.modOn = True
process.SiPixelClusterSource.ladOn = False
process.SiPixelClusterSource.layOn = False
process.SiPixelClusterSource.phiOn = False
process.SiPixelClusterSource.bladeOn = False
process.SiPixelClusterSource.diskOn = False
process.SiPixelClusterSource.ringOn = False
process.SiPixelRecHitSource.saveFile = False
process.SiPixelRecHitSource.modOn = True
process.SiPixelRecHitSource.ladOn = False
process.SiPixelRecHitSource.layOn = False
process.SiPixelRecHitSource.phiOn = False
process.SiPixelRecHitSource.bladeOn = False
process.SiPixelRecHitSource.ringOn = False
process.SiPixelRecHitSource.diskOn = False
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Pixel'
process.dqmSaver.dirName = '.'
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
process.siPixelIsAliveCalibration.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelSCurveAnalysis.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelGainCalibrationAnalysis.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelErrorsDigisToCalibDigis.SiPixelProducerLabelTag = 'siPixelCalibDigis'
process.siPixelIsAliveCalibration.saveFile = False
process.siPixelGainCalibrationAnalysis.saveFile = False
process.siPixelSCurveAnalysis.saveFile = False
process.siPixelErrorsDigisToCalibDigis.saveFile=False
