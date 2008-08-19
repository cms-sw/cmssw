import FWCore.ParameterSet.Config as cms

process = cms.Process("SIPIXELDQM")
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

process.load("CalibTracker.SiPixelGainCalibration.SiPixelCalibDigiProducer_cfi")

process.load("CalibTracker.SiPixelSCurveCalibration.SiPixelSCurveCalibrationAnalysis_cfi")

process.load("CalibTracker.SiPixelIsAliveCalibration.SiPixelIsAliveCalibration_cfi")

process.load("CalibTracker.SiPixelGainCalibration.SiPixelGainCalibrationAnalysis_cfi")

CALIBfrom CalibTracker.Configuration.SiPixelGain.SiPixelGain_Fake_cff import *

CALIBfrom CalibTracker.Configuration.SiPixelLorentzAngle.SiPixelLorentzAngle_Fake_cff import *

CALIBfrom CondCore.DBCommon.CondDBCommon_cfi import *
CALIBCondDBCommon.connect = "oracle://cms_orcoff_prep/CMS_COND_PIXEL_COMM_21X"
CALIBCondDBCommon.DBParameters.authenticationPath = "/afs/cern.ch/cms/DB/conddb"

CALIBprocess.db_pixel_source = cms.ESSource("PoolDBESSource",
CALIB                                        CondDBCommon,
CALIB                                        globaltag = cms.string("PIXELCALIB_01::TypeGLOBALCALIB"),
CALIB                                        BlobStreamerName = cms.untracked.string("TBufferBlobStreamingService")
CALIB)

CALIBprocess.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

PHYSprocess.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

PHYSprocess.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
PHYSprocess.GlobalTag.globaltag = "CRZT210_V1P::All"
PHYSprocess.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


process.source = cms.Source("SOURCETYPE",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/chiochia/cmssw/Muon_FullValidation_150pre3.root')
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
    ONEPARAM
    TWOPARAM                       
    fileNames = cms.untracked.vstring('FILENAME'
				      )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis', 
        'SiPixelRawDataErrorSource', 
        'SiPixelDigiSource', 
        'SiPixelClusterSource', 
        'SiPixelRecHitSource', 
        'sipixelEDAClient'),
    TEXTFILE = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('TEXTFILE')
)

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.sipixelEDAClient = cms.EDFilter("SiPixelEDAClient",
    FileSaveFrequency = cms.untracked.int32(50),
    StaticUpdateFrequency = cms.untracked.int32(10)
)

process.qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.Reco = cms.Sequence(process.siPixelDigis*process.siPixelClusters)
process.RAWmonitor = cms.Sequence(process.SiPixelRawDataErrorSource)
process.DIGImonitor = cms.Sequence(process.SiPixelDigiSource)
process.CLUmonitor = cms.Sequence(process.SiPixelClusterSource)
process.HITmonitor = cms.Sequence(process.SiPixelRecHitSource)
process.DQMmodules = cms.Sequence(process.qTester*process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.DQMmodules*CLUSPOTRECSPOTCDSPOTSCURVESPOTGAINSPOTPIXELSPOTRAWMONSPOTDIGMONSPOTCLUMONSPOTRECMONSPOTprocess.sipixelEDAClient)
#process.p = cms.Path(process.DQMmodules*process.DIGImonitor*process.sipixelEDAClient)
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
process.siPixelIsAliveCalibration.saveFile = False
process.siPixelGainCalibrationAnalysis.saveFile = False
process.siPixelSCurveAnalysis.saveFile = False
