import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.HLTEvF.HLTMonitor_MuonDQM_cff")
process.load("DQM.HLTEvF.HLTMonitorClient_cff")

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'CRAFT_V14P::All'
process.prefer("GlobalTag")

process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    #input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugVebosity = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
    '/store/data/Commissioning09/Monitor/RAW/v1/000/082/380/021C1449-4D2F-DE11-95CB-0019B9F72CE5.root'
    #'file:/tmp/wteo/BA6DA407-A985-DD11-A0D8-000423D9A2AE.root',
    #'file:/tmp/wteo/52BAB7DF-0487-DD11-95A3-000423D9989E.root'
    #'file:/tmp/hdyoo/sample/data/CRUZET4_V6P/CRUZET4_V6P_doubleMuonPath_v1_0.root',
    #'file:/tmp/hdyoo/sample/data/CRUZET4_V6P/CRUZET4_V6P_doubleMuonPath_v1_1.root'
    #'file:/tmp/hdyoo/sample/data/CRUZET3/CRUZET3.root'
    #'file:/tmp/hdyoo/sample/data/CRUZET4_V6P/CRUZET4_V6P_multiCosmicMuon_v1.root'
    #'file:/tmp/hdyoo/sample/MC/CMSSW_2_1_9/RelValSingleMuPt100.root'
    #'file:/tmp/hdyoo/sample/data/PreCRUZET4Test1/PreCRUZET4Test1.root'
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
# 
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
