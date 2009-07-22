import FWCore.ParameterSet.Config as cms

process = cms.Process("SIPIXELDQM")

##----## Geometry and other global parameters:
#process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.MagneticField_0T_cff")
#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

##----## Reco:
process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True
#process.load("Configuration.StandardSequences.RawToDigi_cff")

#process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

#process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
#process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")

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

##new##
#process.load("Configuration.StandardSequences.Simulation_cff")



##----## Central DQM:
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Pixel'
process.dqmSaver.dirName = '/tmp/merkelp/'
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
process.qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml'),
    prescaleFactor = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    testInEventloop = cms.untracked.bool(False),
    qtestOnEndRun = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    verboseQT = cms.untracked.bool(False)
)

##----## Pixel DQM P5/OFFLINE:
process.load("DQM.SiPixelCommon.SiPixelOfflineDQM_source_cff")
process.load("DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff")
#process.load("DQM.SiPixelCommon.SiPixelP5DQM_source_cff")
#process.load("DQM.SiPixelCommon.SiPixelP5DQM_client_cff")
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
process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)


##----## Global tag and input data:
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRZT210_V1.db"
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_31X_V4P::All"
## this next line is needed for xdaq running outside P5 network:
#process.SiteLocalConfigService = cms.Service("SiteLocalConfigService")


process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/chiochia/cmssw/Muon_FullValidation_150pre3.root')
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
    fileNames = cms.untracked.vstring(
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/066/668/ECBAB6B1-519C-DD11-BBB5-000423D94E70.root'
	#'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root'
	#'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/00BAAF73-52A5-DD11-9351-001D09F23A84.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/00FD70FC-3DA5-DD11-A7BE-000423D9A2AE.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/00FED19D-03A5-DD11-83E4-001D09F241B9.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/00FFFBAD-0EA5-DD11-BADC-001D09F24498.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/0211E182-1DA5-DD11-ADF6-001D09F2960F.root'

        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/FCC69A9F-92A9-DD11-A11B-0030487D0D3A.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/FC99CAD7-9DA9-DD11-A329-000423D98EA8.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/FC42A8F3-91A9-DD11-928A-000423D99896.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/FAEE63A4-99A9-DD11-A13C-001D09F254CE.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/FA5456BD-9BA9-DD11-9BEC-001617E30CA4.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/F8E22217-94A9-DD11-AD05-000423D99EEE.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/F885F252-9FA9-DD11-9CEE-000423DD2F34.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/F82CF617-94A9-DD11-BDA5-000423D9997E.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/F65C4BDB-9DA9-DD11-9945-0030487A3C9A.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/F622722A-8FA9-DD11-91DA-000423D6B2D8.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/F4F0E0D3-96A9-DD11-9182-0030487A322E.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/F23B94CB-A2A9-DD11-8FC2-001617E30E2C.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/EAE86207-94A9-DD11-A851-0016177CA7A0.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/E47DFDB2-A0A9-DD11-AE40-000423D60FF6.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/E2CB71BD-94A9-DD11-920D-000423D944DC.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/E0F5E89D-92A9-DD11-8224-0030487C5CFA.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/DE15DFA0-99A9-DD11-94A4-000423D991F0.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/DCED14E1-9DA9-DD11-9B78-001617C3B706.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/DA2771F2-98A9-DD11-9E32-0030487A322E.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/D89DC864-A1A9-DD11-87E3-000423D98844.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/D4A9799A-9EA9-DD11-B823-000423D98E6C.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/D231A4D1-9DA9-DD11-9EF9-000423D6CAF2.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/D00E48B4-A0A9-DD11-916D-001D09F2447F.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/CC13AE8D-97A9-DD11-BAD9-00304879FA4A.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/CA7B876C-9CA9-DD11-B143-000423D9A212.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/CA03D449-9FA9-DD11-946F-001617C3B6E8.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/C8B88288-90A9-DD11-B16C-000423D98834.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/C4D0D970-9CA9-DD11-85CE-000423D6CA42.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/C46E1F53-9FA9-DD11-9D04-001D09F24498.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/C2D160D9-8FA9-DD11-8C52-000423D985B0.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/046/C2980351-9FA9-DD11-89B5-00304879FBB2.root'

        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/FECCF15C-4872-DE11-BDB2-000423D944F8.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/FE0DF038-4672-DE11-B22B-001D09F295FB.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/FABEA7C6-4972-DE11-BA80-0030487C6062.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/F88DE49D-4072-DE11-8ECC-0019B9F730D2.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/F63F1560-4872-DE11-84DD-0019B9F72D71.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/F48138CE-4472-DE11-A947-001D09F231B0.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/F24BB5ED-3F72-DE11-9311-001D09F2B30B.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/F0A5C462-4F72-DE11-B6F3-001D09F2305C.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/EC51527A-3E72-DE11-AB2E-001D09F29619.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/E6D5B5EA-4B72-DE11-A417-0030487BC68E.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/E2EAA9CE-4472-DE11-A6CA-001D09F282F5.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/E2A9FAAA-5372-DE11-89A6-000423D99CEE.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/E26879FF-4D72-DE11-AB35-001D09F292D1.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/E07B3D2D-4B72-DE11-8CBE-001D09F253FC.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/E0659D7C-3E72-DE11-895E-001D09F2462D.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/DE1F75AF-5372-DE11-B70A-000423D9880C.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/DAAA030C-5572-DE11-A224-001D09F24493.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/D8F8924E-3C72-DE11-9444-0030487A1FEC.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/D8E91107-5572-DE11-B268-000423D94524.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/D4C7B9CF-5072-DE11-8FD7-001D09F282F5.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/CC41491F-5072-DE11-84E6-001D09F29146.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/CAD54B6A-4372-DE11-ADF7-001D09F25460.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/CA1ACD44-4D72-DE11-86D1-000423D98800.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/C6F6C255-6972-DE11-AB0F-001D09F23A02.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/C4B50D00-4E72-DE11-8DF5-001D09F251B8.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/C47195C7-3D72-DE11-86DF-000423D987E0.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/C0CA162F-4B72-DE11-ABE1-001D09F2516D.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/C07FD3C8-4472-DE11-B5FC-000423D944FC.root',
        '/store/data/Commissioning09/Cosmics/RAW/v3/000/106/019/BA4C1EBC-4E72-DE11-8BD4-001D09F2905B.root'
	    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

##----## Sequences and Paths:
#process.Reco = cms.Sequence(process.siPixelRecHits)
process.Reco = cms.Sequence(process.siPixelDigis*process.siPixelClusters*process.siPixelRecHits)
#process.RecoStrips = cms.Sequence(process.siStripDigis*process.siStripClusters)
process.RAWmonitor = cms.Sequence(process.SiPixelRawDataErrorSource)
process.DIGImonitor = cms.Sequence(process.SiPixelDigiSource)
process.CLUmonitor = cms.Sequence(process.SiPixelClusterSource)
process.HITmonitor = cms.Sequence(process.SiPixelRecHitSource)
process.DQMmodules = cms.Sequence(process.qTester*process.dqmEnv*process.dqmSaver)
process.CERTmodules = cms.Sequence(process.sipixelDaqInfo*process.sipixelDcsInfo*process.sipixelCertification)
#process.siPixelLocalReco = cms.Sequence(process.siPixelRecHits) 
#process.siStripLocalReco = cms.Sequence(process.siStripMatchedRecHits)
#process.trackerLocalReco = cms.Sequence(process.siPixelLocalReco*process.siStripLocalReco)
#process.trackReconstruction = cms.Sequence(process.trackerLocalReco*process.offlineBeamSpot*process.recopixelvertexing*process.tracksP5) #*process.rstracks 
#process.monitorTrack = cms.Sequence(process.SiPixelTrackResidualSource)
#process.monitors = cms.Sequence(process.SiPixelRawDataErrorSource*process.SiPixelDigiSource*process.SiPixelClusterSource*process.SiPixelRecHitSource*process.SiPixelTrackResidualSource)

#put proces.dump in the path where you want to print all event content
#process.dump=cms.EDAnalyzer('EventContentAnalyzer')

#process.p = cms.Path(process.Reco*process.RecoStrips*process.trackReconstruction*process.dqmEnv*process.siPixelP5DQM_cosmics_source*process.qTester*process.PixelP5DQMClientWithDataCertification*process.dqmSaver)
#process.pathTrack = cms.Path(process.trackReconstruction*process.DQMmodules*process.monitors*process.sipixelEDAClient) 
#process.p = cms.Path(process.Reco*process.dqmEnv*process.siPixelOfflineDQM_source_woTrack*process.PixelOfflineDQMClientWithDataCertification*process.dqmSaver)
process.p = cms.Path(process.Reco*process.dqmEnv*process.siPixelOfflineDQM_source_woTrack*process.qTester*process.PixelOfflineDQMClientWithDataCertification*process.dqmSaver)
#process.p = cms.Path(process.siPixelDigis*process.siPixelClusters*process.trackReconstruction*process.dqmEnv*process.siPixelP5DQM_cosmics_source*process.PixelP5DQMClient*process.dqmSaver)

#process.p = cms.Path( process.siPixelDigis * process.dqmEnv*process.SiPixelDigiSource*process.PixelP5DQMClient*process.dqmSaver)
#process.p = cms.Path(process.siPixelDigis*process.dqmEnv*process.siPixelP5DQM_cosmics_source*process.qTester*process.PixelP5DQMClientWithDataCertification*process.dqmSaver)
