import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.HLTEvF.HLTMonitor_MuonDQM_cff")
process.load("DQM.HLTEvF.HLTMonitorClient_cff")

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'IDEAL_30X::All'
process.prefer("GlobalTag")

process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugVebosity = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
    'file:/tmp/hdyoo/28F5E252-180A-DE11-87E2-000423D94A20.root'
    )
)


#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring('file:/nfshome0/lorenzo/305FBD08-D78F-DD11-A1D6-001617C3B65A.root')
#   fileNames = cms.untracked.vstring(
#        'file:/cmsdisk1/lookarea_SM/GlobalMW40.00064245.0001.HLTDEBUG.storageManager.01.0000.dat',
#        'file:/cmsdisk1/lookarea_SM/GlobalMW40.00064210.0001.HLTDEBUG.storageManager.03.0000.dat'
#  )
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
# process.MessageLogger.categories = ['hltResults']
# #process.MessageLogger.destinations = ['cout', 'detailedInfo', 'critical']
# process.MessageLogger.cout = cms.untracked.PSet(
#     #threshold = cms.untracked.string('ERROR'),
#     threshold = cms.untracked.string('INFO'),
#     INFO = cms.untracked.PSet(
#      limit = cms.untracked.int32(-1)
#     ),
#     threshold = cms.untracked.string('DEBUG'),
#     DEBUG = cms.untracked.PSet(
#     limit = cms.untracked.int32(-1) ## DEBUG, all messages
#     )
#     )
# process.MessageLogger.categories = ['Status', 'Parameter']
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
