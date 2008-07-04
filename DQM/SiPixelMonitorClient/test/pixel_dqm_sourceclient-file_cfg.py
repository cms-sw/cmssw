import FWCore.ParameterSet.Config as cms

process = cms.Process("SIPIXELDQM")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")

process.load("CalibTracker.Configuration.Tracker_FakeConditions_cff")

process.load("CalibTracker.SiPixelESProducers.SiPixelFakeGainESSource_cfi")

process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi")

process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")

process.load("DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi")

process.load("DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/chiochia/cmssw/Muon_FullValidation_150pre3.root')
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
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cout')
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

process.Reco = cms.Sequence(process.siPixelClusters)
process.RAWmonitor = cms.Sequence(process.SiPixelRawDataErrorSource)
process.DIGImonitor = cms.Sequence(process.SiPixelDigiSource)
process.CLUmonitor = cms.Sequence(process.SiPixelClusterSource)
process.HITmonitor = cms.Sequence(process.SiPixelRecHitSource)
process.DQMmodules = cms.Sequence(process.qTester*process.dqmEnv*process.dqmSaver)
#process.p = cms.Path(process.Reco*process.DQMmodules*process.DIGImonitor*process.CLUmonitor*process.HITmonitor*process.sipixelEDAClient)
process.p = cms.Path(process.DQMmodules*process.DIGImonitor*process.sipixelEDAClient)
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True
process.SiPixelRawDataErrorSource.saveFile = False
process.SiPixelRawDataErrorSource.isPIB = False
process.SiPixelRawDataErrorSource.slowDown = False
process.SiPixelDigiSource.saveFile = False
process.SiPixelDigiSource.isPIB = False
process.SiPixelDigiSource.slowDown = False
process.SiPixelDigiSource.modOn = False
process.SiPixelDigiSource.ladOn = True
process.SiPixelDigiSource.layOn = True
process.SiPixelDigiSource.phiOn = True
process.SiPixelDigiSource.bladeOn = True
process.SiPixelDigiSource.diskOn = True
process.SiPixelDigiSource.ringOn = True
process.SiPixelClusterSource.saveFile = False
process.SiPixelClusterSource.modOn = False
process.SiPixelClusterSource.ladOn = True
process.SiPixelClusterSource.layOn = True
process.SiPixelClusterSource.phiOn = True
process.SiPixelClusterSource.bladeOn = True
process.SiPixelClusterSource.diskOn = True
process.SiPixelClusterSource.ringOn = True
process.SiPixelRecHitSource.saveFile = False
process.SiPixelRecHitSource.modOn = False
process.SiPixelRecHitSource.ladOn = True
process.SiPixelRecHitSource.layOn = True
process.SiPixelRecHitSource.phiOn = True
process.SiPixelRecHitSource.bladeOn = True
process.SiPixelRecHitSource.ringOn = True
process.SiPixelRecHitSource.diskOn = True
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Pixel'
process.dqmSaver.saveByLumiSection = 1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

