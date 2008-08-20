import FWCore.ParameterSet.Config as cms

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

process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi")
process.SiPixelRawDataErrorSource.saveFile = False
process.SiPixelRawDataErrorSource.isPIB = False
process.SiPixelRawDataErrorSource.slowDown = False

process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")
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

process.load("DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi")
process.SiPixelClusterSource.saveFile = False
process.SiPixelClusterSource.isPIB = False
process.SiPixelClusterSource.modOn = True
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
process.SiPixelRecHitSource.ladOn = False
process.SiPixelRecHitSource.layOn = False
process.SiPixelRecHitSource.phiOn = False
process.SiPixelRecHitSource.bladeOn = False
process.SiPixelRecHitSource.ringOn = False
process.SiPixelRecHitSource.diskOn = False

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRZT210_V1.db"
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V1::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/chiochia/cmssw/Muon_FullValidation_150pre3.root')
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/2CB57036-4C6A-DD11-9A04-000423D94990.root',
    #                                  'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/7EB80A4C-4C6A-DD11-A589-000423D987E0.root',
    #                                  'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/B0319435-4C6A-DD11-9B18-000423D952C0.root',
    #                                  'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/B428FF51-4C6A-DD11-87D8-000423D986C4.root',
    #                                  'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/DEAB1B33-4C6A-DD11-B83D-000423D99BF2.root'
#				      )
#    fileNames = cms.untracked.vstring(
                                      #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/741/2267B14D-4C6A-DD11-BEFE-001617C3B70E.root',
                                      #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/741/2A16B54C-4C6A-DD11-9612-001617C3B6E2.root',
                                      #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/741/32B8FC50-4C6A-DD11-BCDA-000423D944DC.root'
                                      #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/741/3EC4E22A-4C6A-DD11-B533-000423D94C68.root',
                                      #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/741/48232F48-4C6A-DD11-84AE-000423D996B4.root',
                                      #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/741/6016DF32-4C6A-DD11-AE83-000423D94A04.root'
	#			      )
    fileNames = cms.untracked.vstring(
                                      'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/774/2260AF5F-356E-DD11-99F3-000423D6C8E6.root',
                                      'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/774/2260AF5F-356E-DD11-99F3-000423D6C8E6.root',
                                      'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/774/2260AF5F-356E-DD11-99F3-000423D6C8E6.root',
                                      'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/774/2260AF5F-356E-DD11-99F3-000423D6C8E6.root',
                                      'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/774/2260AF5F-356E-DD11-99F3-000423D6C8E6.root',
                                      'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/774/2260AF5F-356E-DD11-99F3-000423D6C8E6.root'
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
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
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

process.Reco = cms.Sequence(process.siPixelDigis*process.siPixelClusters*siPixelRecHits)
process.RAWmonitor = cms.Sequence(process.SiPixelRawDataErrorSource)
process.DIGImonitor = cms.Sequence(process.SiPixelDigiSource)
process.CLUmonitor = cms.Sequence(process.SiPixelClusterSource)
process.HITmonitor = cms.Sequence(process.SiPixelRecHitSource)
#process.DQMmodules = cms.Sequence(process.qTester*process.dqmEnv*process.dqmSaver)
process.DQMmodules = cms.Sequence(process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.Reco*process.DQMmodules*process.RAWmonitor*process.DIGImonitor*process.CLUmonitor*process.HITmonitor*process.sipixelEDAClient)
#process.p = cms.Path(process.DQMmodules*process.DIGImonitor*process.sipixelEDAClient)
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Pixel'
process.dqmSaver.dirName = '/tmp/merkelp/'
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

