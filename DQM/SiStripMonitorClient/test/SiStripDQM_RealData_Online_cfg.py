import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineRealData")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiStripZeroSuppression', 
        'SiStripMonitorDigi', 
        'SiStripMonitorCluster', 
        'SiStripMonitorTrackSim', 
        'MonitorTrackResidualsSim',
	'Timing'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)
#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring("Timing"),
#    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
#    destinations = cms.untracked.vstring('cout')
#    #destinations = cms.untracked.vstring("detailedInfo"),
#    #detailedInfo = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))
#)
#process.Timing = cms.Service("Timing")
#-------------------------------------------------
# Magnetic Field
#-------------------------------------------------
# 0T field
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")
# 3.8T field 
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.prefer("VolumeBasedMagneticFieldESProducer")

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT09_R_V3::All"

#If Frontier is used in xdaq environment use the following service
#process.SiteLocalConfigService = cms.Service("SiteLocalConfigService")
#-----------------------
# Reconstruction Modules
#-----------------------
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.RecoForDQM_RealData_Cosmics = cms.Sequence(
	process.siPixelDigis*
	process.siStripDigis*
	process.trackerlocalreco*
	process.offlineBeamSpot*
	process.ctftracksP5
)

#--------------------------
# DQM
#--------------------------
## Source
process.load("DQM.SiStripMonitorClient.SiStripSourceConfigP5_cff")

process.SiStripSources_Common = cms.Sequence(
	#process.siStripFEDCheck*
	process.siStripFEDMonitor*
	process.SiStripMonitorDigi*
	process.SiStripMonitorClusterReal
)
process.SiStripSources_Cosmics = cms.Sequence(
	process.SiStripMonitorTrack_ckf*
	process.MonitorTrackResiduals_ckf*
	process.TrackMon_ckf
)

## Client
process.load("DQM.SiStripMonitorClient.SiStripClientConfigP5_cff")
process.SiStripAnalyser.StaticUpdateFrequency = cms.untracked.int32(-1)
process.SiStripAnalyser.GlobalStatusFilling = cms.untracked.int32(2)
process.SiStripAnalyser.TkMapCreationFrequency = cms.untracked.int32(-1)
process.SiStripAnalyser.ShiftReportFrequency = cms.untracked.int32(-1)
process.SiStripAnalyser.PrintFaultyModuleList = cms.untracked.bool(True)

process.SiStripClients = cms.Sequence(
        process.SiStripAnalyser
)
#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.DQM.filter = '^SiStrip(/[^/]+){0,5}$'

process.load("DQMServices.Components.DQMEnvironment_cfi")

#--------------------------
# Quality Test
#--------------------------
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    getQualityTestsFromFile = cms.untracked.bool(True)
)
#----------------------------
# DQM Playback Environment
#-----------------------------
process.load("DQM.Integration.test.environment_playback_cfi")
process.dqmEnv.subSystemFolder    = "SiStrip"
process.dqmSaver.saveByMinute     = 120
process.dqmSaver.dirName  = cms.untracked.string(".")
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.dqmEnvTr = DQMEDAnalyzer('DQMEventInfo',
                       subSystemFolder = cms.untracked.string('Tracking'),
                       eventRateWindow = cms.untracked.double(0.5),
                       eventInfoFolder = cms.untracked.string('EventInfo')
                   )

process.DQMCommon = cms.Sequence(
	process.qTester*
        process.dqmEnv*
        process.dqmEnvTr*
        process.dqmSaver
)

#----------------------------
# Scheduling
#-----------------------------
process.p = cms.Path(
	process.RecoForDQM_RealData_Cosmics*
	process.DQMCommon*
	process.SiStripSources_Common*
	process.SiStripSources_Cosmics*
	process.SiStripClients
)

process.AdaptorConfig = cms.Service("AdaptorConfig")

#-------------------------
# Input Events
#-------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
## CRAFT09
       '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/998/001404E1-0F8A-DE11-ADB3-000423D99EEE.root',
       '/store/data/CRAFT09/Cosmics/RAW/v1/000/110/998/002174A8-E989-DE11-8B4D-000423D6CA42.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
