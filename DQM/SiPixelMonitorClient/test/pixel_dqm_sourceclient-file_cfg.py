import FWCore.ParameterSet.Config as cms

process = cms.Process("SIPIXELDQM")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

##process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True

process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")

#process.load("EventFilter.SiStripRawToDigi.SiStripRawToDigis_standard_cff")
#process.siStripDigis.ProductLabel = 'source'

#process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi")
#process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi")
#process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi")
#process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi")
#process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")

#process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
#process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
#process.load("RecoTracker.Configuration.RecoTrackerP5_cff")
  







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

# process.load("DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_cfi")
# process.SiPixelTrackResidualSource.TrackCandidateProducer = 'ckfTrackCandidatesP5'
# process.SiPixelTrackResidualSource.TrackProducer = 'ALCARECOTkAlCosmicsCosmicTF0T'
# process.SiPixelTrackResidualSource.debug = True
# process.SiPixelTrackResidualSource.outputFile = '/tmp/merkelp/Pixel_DQM_TrackResidual_cosmictrackfinderP5.root'


process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRZT210_V1.db"
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V5P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/chiochia/cmssw/Muon_FullValidation_150pre3.root')
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
    fileNames = cms.untracked.vstring(
    #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/2CB57036-4C6A-DD11-9A04-000423D94990.root',
    #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/7EB80A4C-4C6A-DD11-A589-000423D987E0.root',
    #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/B0319435-4C6A-DD11-9B18-000423D952C0.root',
    #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/B428FF51-4C6A-DD11-87D8-000423D986C4.root',
    #'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/742/DEAB1B33-4C6A-DD11-B83D-000423D99BF2.root'
    #'rfio:/castor/cern.ch/cms/store/cmscaf/alca/alignment/CRUZET4-TkAlCosmics/57553/ALCARECOTkAlCosmics0T_1.root', 
    #rfio:/castor/cern.ch/cms/store/cmscaf/alca/alignment/CRUZET4-TkAlCosmics/57553/ALCARECOTkAlCosmics0T_2.root'
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/0083B2EF-3F7A-DD11-9165-001617C3B77C.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/06B08C98-3E7A-DD11-9F5F-000423D94700.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/082621C9-387A-DD11-9ECB-001617C3B77C.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/304B5906-3F7A-DD11-8E21-000423D98804.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/3E8C90DE-3A7A-DD11-ADB1-000423D6CAF2.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/44E16FAB-3D7A-DD11-8075-000423D94700.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/4654708F-397A-DD11-9D4A-001617C3B5F4.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/4EB8F3B8-397A-DD11-9DD5-000423D9880C.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/5C4429DE-3A7A-DD11-B9B1-001617C3B6C6.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/5E761ABB-387A-DD11-AE3B-000423D6BA18.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/6C2235E0-397A-DD11-BBEE-001617C3B79A.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/7E4EF558-3B7A-DD11-A223-001617C3B6CC.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/86C71864-3B7A-DD11-911B-001617C3B654.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/86DABEE2-3A7A-DD11-B2EB-000423D6CA6E.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/8CAC62C9-377A-DD11-B775-0019DB29C620.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/96498981-397A-DD11-BE92-000423D9870C.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/A0486627-3A7A-DD11-81DD-000423D6BA18.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/A2DDA7A2-3B7A-DD11-A99F-001617C3B6CE.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/C2CA7A8F-377A-DD11-B072-001617E30F56.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/D0573EA4-3C7A-DD11-8B46-0019DB29C5FC.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/F2C36B6E-397A-DD11-8EDD-000423D9880C.root',
        '/store/data/Commissioning08/BeamHalo/RAW/MW36_v1/000/061/070/F6D2D110-3C7A-DD11-8579-001617C3B6DE.root'				      )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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
    EventOffsetForInit = cms.untracked.int32(10),
    ActionOnLumiSection = cms.untracked.bool(False),
    ActionOnRunEnd = cms.untracked.bool(True),
    HighResolutionOccupancy = cms.untracked.bool(True),
    NoiseRateCutValue = cms.untracked.double(0.0001)
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

#process.Reco = cms.Sequence(process.siPixelRecHits)
process.Reco = cms.Sequence(process.siPixelDigis*process.siPixelClusters*process.siPixelRecHits)
process.RAWmonitor = cms.Sequence(process.SiPixelRawDataErrorSource)
process.DIGImonitor = cms.Sequence(process.SiPixelDigiSource)
process.CLUmonitor = cms.Sequence(process.SiPixelClusterSource)
process.HITmonitor = cms.Sequence(process.SiPixelRecHitSource)
process.DQMmodules = cms.Sequence(process.qTester*process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.Reco*process.qTester*process.dqmEnv*process.RAWmonitor*process.DIGImonitor*process.CLUmonitor*process.HITmonitor*process.sipixelEDAClient*process.dqmSaver)
#process.p = cms.Path(process.Reco*process.DQMmodules*process.RAWmonitor*process.DIGImonitor*process.CLUmonitor*process.HITmonitor*process.sipixelEDAClient)
#process.p = cms.Path(process.DQMmodules*process.DIGImonitor*process.sipixelEDAClient)
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Pixel'
process.dqmSaver.dirName = '/tmp/merkelp/'
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

#process.siPixelLocalReco = cms.Sequence(process.siPixelRecHits) 
#process.siStripLocalReco = cms.Sequence(process.siStripMatchedRecHits)
#process.trackerLocalReco = cms.Sequence(process.siPixelLocalReco*process.siStripLocalReco)
#process.trackReconstruction = cms.Sequence(process.trackerLocalReco*process.offlineBeamSpot*process.recopixelvertexing*process.tracksP5) #*process.rstracks 

#process.monitorTrack = cms.Sequence(process.SiPixelTrackResidualSource)
#process.monitors = cms.Sequence(process.SiPixelRawDataErrorSource*process.SiPixelDigiSource*process.SiPixelClusterSource*process.SiPixelRecHitSource*process.SiPixelTrackResidualSource)

#process.pathTrack = cms.Path(process.trackReconstruction*process.DQMmodules*process.monitors*process.sipixelEDAClient) 
