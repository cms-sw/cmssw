import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.HLTEvF.HLTMonitor_cff")
process.load("DQM.HLTEvF.HLTMonitorClient_cff")
process.load("DQM.TrigXMonitor.HLTScalers_cfi")
process.load("DQM.TrigXMonitorClient.HLTScalersClient_cfi")
process.hlts.l1GtData = cms.InputTag("l1GtUnpack","","DQM")
process.hlts.dqmFolder = cms.untracked.string("HLT/HLTScalers_SM")
process.hltsClient.dqmFolder = cms.untracked.string("HLT/HLTScalers_SM")
process.p = cms.EndPath(process.hlts+process.hltsClient)


process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'GR09_E_V3T::All'
process.prefer("GlobalTag")

#SiStrip Local Reco
process.load("DQM.SiStripCommon.TkHistoMap_cff")

process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )

process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(-1)
#)

process.load("DQM.HLTEvF.118395_cfi")

#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/data/Commissioning09/Cosmics/RAW/v3/000/118/967/006BA396-9378-DE11-BBC8-000423D944FC.root')
#fileNames = cms.untracked.vstring('/store/data/Commissioning09/Cosmics/RAW/v3/000/118/967/425F94EF-ACC5-DE11-841C-000423D6B5C4.root',
#'/store/data/Commissioning09/Cosmics/RAW/v3/000/118/967/604806AF-A1C5-DE11-AC49-001D09F28D54.root',
#'/store/data/Commissioning09/Cosmics/RAW/v3/000/118/967/68A2080D-A3C5-DE11-AA14-0030487C6090.root',
#'/store/data/Commissioning09/Cosmics/RAW/v3/000/118/967/7AB4D963-A2C5-DE11-926D-003048D2BE08.root',
#'/store/data/Commissioning09/Cosmics/RAW/v3/000/118/967/7CF1EA63-A2C5-DE11-BB4C-001617E30CD4.root',
#'/store/data/Commissioning09/Cosmics/RAW/v3/000/118/967/D42C1867-A2C5-DE11-98DC-003048D2C108.root')
#)
#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring(
#                                      'file:/tmp/lorenzo/RunPrep09.00119406.0026.A.storageManager.05.0000.dat'
#				      )
#)				      




#process.p = cms.EndPath(process.hlts+process.hltsClient)


process.pp = cms.Path(process.dqmEnv+process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = 'srv-c2d05-12'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
#process.hltResults.plotAll = True
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

# # Message Logger
# process.load("FWCore.MessageService.MessageLogger_cfi")
# #
# process.MessageLogger.destinations = ['cout', 'detailedInfo', 'critical']
# process.MessageLogger.cout = cms.untracked.PSet(
#     #threshold = cms.untracked.string('ERROR'),
#     #threshold = cms.untracked.string('INFO'),
#     #INFO = cms.untracked.PSet(
#     # limit = cms.untracked.int32(-1)
#     #)#,
#     threshold = cms.untracked.string('DEBUG'),
#     DEBUG = cms.untracked.PSet(
#     limit = cms.untracked.int32(-1) ## DEBUG, all messages
#     )
#     )
# 
# # copy stdout to a file
# process.MessageLogger.detailedInfo = process.MessageLogger.cout
# process.MessageLogger.debugModules = ['hltResults']
# process.MessageLogger.critical = cms.untracked.PSet(
#     threshold = cms.untracked.string('ERROR'),
#     #threshold = cms.untracked.string('INFO'),
#     #INFO = cms.untracked.PSet(
#     # limit = cms.untracked.int32(-1)
#     #)#,
#     #threshold = cms.untracked.string('DEBUG'),
#     ERROR = cms.untracked.PSet(
#     limit = cms.untracked.int32(-1) ## all messages
#     )
#     )
# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )
