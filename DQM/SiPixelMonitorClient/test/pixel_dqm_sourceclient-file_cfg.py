import FWCore.ParameterSet.Config as cms

process = cms.Process("SIPIXELDQM")

##----## Geometry and other global parameters:
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
###process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

##----## Reco:
process.load("Configuration.StandardSequences.Reconstruction_cff")

#process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True

process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

#process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
process.load("EventFilter.SiStripRawToDigi.SiStripRawToDigis_standard_cff")
process.siStripDigis.ProductLabel = 'source'

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")


##----## Central DQM:
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Pixel'
process.dqmSaver.dirName = '/tmp/merkelp/'
process.dqmSaver.saveByLumiSection = 1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
process.qTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelMonitorClient/test/sipixel_tier0_qualitytest.xml'),
    prescaleFactor = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    testInEventloop = cms.untracked.bool(False),
    qtestOnEndRun = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    verboseQT = cms.untracked.bool(False)
)

##----## Pixel DQM P5/OFFLINE:
#process.load("DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff")
#process.load("DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff")
process.load("DQM.SiPixelCommon.SiPixelP5DQM_source_cff")
process.load("DQM.SiPixelCommon.SiPixelP5DQM_client_cff")
## the following sequences are declared therein:
## siPixelOfflineDQM_source, siPixelOfflineDQM_cosmics_source, siPixelOfflineDQM_source_woTrack
## PixelOfflineDQMClient, PixelOfflineDQMClientWithDataCertification
## siPixelP5DQM_source, siPixelP5DQM_cosmics_source, siPixelP5DQM_source_woTrack
## PixelP5DQMClient, PixelP5DQMClientWithDataCertification



##----## Other stuff:
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
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

##----## Global tag and input data:
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRZT210_V1.db"
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
###process.GlobalTag.globaltag = "GR09_R_V4::All"
process.GlobalTag.globaltag = "CRAFT09_R_V9::All"

process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/chiochia/cmssw/Muon_FullValidation_150pre3.root')
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
    fileNames = cms.untracked.vstring(
    #'/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/124/275/ECC2F9D5-F7E9-DE11-BF99-001D09F282F5.root',
    #'/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/124/275/DE9A0B4E-F4E9-DE11-BF54-001D09F251CC.root',
    #'/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/124/275/7AD672E3-F2E9-DE11-8173-001D09F24498.root',
    #'/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/124/275/42B6AB0A-F5E9-DE11-8A92-001D09F2546F.root',
    #'/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/124/275/228BA375-F6E9-DE11-8D89-000423D6A6F4.root',
    #'/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/124/275/026B6140-F9E9-DE11-A392-001D09F28755.root'
    '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/275/FA12DE16-FCE9-DE11-8FFE-001D09F24DA8.root',
    '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/275/DE3F1AC9-F7E9-DE11-85C9-001D09F24303.root',
    '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/275/C83B2F2C-FEE9-DE11-8F2E-001D09F24934.root',
    '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/275/A8E551ED-F9E9-DE11-A59E-001D09F29849.root'
	    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

##----## Sequences and Paths:
process.LocalReco = cms.Sequence(process.siPixelDigis*process.siPixelClusters*process.siPixelRecHits)
process.TrackReco = cms.Sequence(process.siStripDigis*process.offlineBeamSpot*process.trackerlocalreco*process.recopixelvertexing*process.ckftracks)
process.DQMmodules = cms.Sequence(process.qTester*process.dqmEnv*process.dqmSaver)

#put proces.dump in the path where you want to print all event content
#process.dump=cms.EDAnalyzer('EventContentAnalyzer')

process.p = cms.Path(
		     process.LocalReco*
		     process.TrackReco*
		     process.DQMmodules*
		     process.siPixelP5DQM_source*
		     process.PixelP5DQMClientWithDataCertification
		    )
