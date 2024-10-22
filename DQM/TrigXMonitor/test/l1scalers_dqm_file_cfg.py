import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

#process.load("EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.l1GtUnpack.DaqGtInputTag = 'source'

process.load("DQM.TrigXMonitor.L1Scalers_cfi")
process.load("DQM.TrigXMonitorClient.L1TScalersClient_cfi")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(100)
    input = cms.untracked.int32(-1)
)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = "GR10_P_V5::All"

process.source = cms.Source("PoolSource",
#    debugVerbosity = cms.untracked.uint32(1),
#    debugVebosity = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
#        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/FC1AFFE1-4609-DE11-928D-001D09F29597.root',
#        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/FE143197-2C09-DE11-8CE8-000423D98844.root'

#	'/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/00D22895-3109-DE11-A8A8-0030487A3C9A.root'
#       '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/02427DDC-4609-DE11-B6F4-001D09F26C5C.root'
#        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/04329D1E-4D09-DE11-9434-001617C3B78C.root'
#        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/04710543-4A09-DE11-90DA-000423D174FE.root'
 #       '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/06685B48-3C09-DE11-B465-001617C3B5E4.root',
 #       '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/06818771-2D09-DE11-BB50-0019B9F72CC2.root',
 #       '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/08C598DA-3009-DE11-8ABA-001617C3B6DC.root',
 #       '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/08EDE497-3109-DE11-B76A-00304879FA4C.root',
 #       '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/0CEE5190-4E09-DE11-8729-001617E30F48.root',
 #       '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/0E866721-5909-DE11-84F7-001D09F231C9.root',
 #       '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/101518B2-3609-DE11-A3C5-001D09F2A690.root',



#        '/store/data/Commissioning09/Cosmics/RAW/v1/000/081/156/46774F48-8324-DE11-8255-000423D996B4.root',
#        '/store/data/Commissioning09/Cosmics/RAW/v1/000/081/156/C4042044-8324-DE11-B07D-000423D944FC.root'

#        '/store/data/Commissioning09/Cosmics/RAW/v1/000/082/548/0617B763-4930-DE11-BCB4-000423D9997E.root',
#        '/store/data/Commissioning09/Cosmics/RAW/v1/000/082/548/08758193-4930-DE11-B023-000423D99996.root'
#	'file:/tmp/0426DEEE-F937-DF11-8C7F-001D09F29538.root'
	#'file:/tmp/0AD63868-0138-DF11-9F74-000423D95220.root'
        'file:/tmp/00661694-BA67-DF11-810D-000423D98C20.root'
    )
#    skipEvents = cms.untracked.uint32(5000)
)


#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring('file:/nfshome0/lorenzo/305FBD08-D78F-DD11-A1D6-001617C3B65A.root')
#   fileNames = cms.untracked.vstring(
#        'file:/cmsdisk1/lookarea_SM/GlobalMW40.00064245.0001.HLTDEBUG.storageManager.01.0000.dat',
#        'file:/cmsdisk1/lookarea_SM/GlobalMW40.00064210.0001.HLTDEBUG.storageManager.03.0000.dat'
#  )
#)


#process.p = cms.EndPath(process.hlts+process.hltsClient)

#process.assist = cms.Path(process.scalersRawToDigi)
#process.assist=cms.Path(process.l1GtUnpack)

process.p = cms.EndPath(process.l1GtUnpack*process.l1s + process.l1tsClient)

process.pp = cms.Path(process.dqmEnv+process.dqmSaver)
process.DQMStore.verbose = 0
#process.DQM.collectorHost = 'srv-c2d05-12'
process.DQM.collectorHost = 'lxplus255'
process.DQM.collectorPort = 9190

process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'L1T'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

# Message Logger
#process.load("FWCore.MessageService.MessageLogger_cfi")
#
#process.MessageLogger.destinations = ['cout', 'detailedInfo', 'critical']
#process.MessageLogger.cout = cms.untracked.PSet(
     #threshold = cms.untracked.string('ERROR'),
#    threshold = cms.untracked.string('INFO'),
#    INFO = cms.untracked.PSet(
#      limit = cms.untracked.int32(-1)
#    ),
    #threshold = cms.untracked.string('DEBUG'),
    #DEBUG = cms.untracked.PSet(
    #   limit = cms.untracked.int32(-1) ## DEBUG, all messages
    #)
#  )

# 
# # copy stdout to a file
process.MessageLogger.detailedInfo = process.MessageLogger.cout
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

process.MessageLogger.cerr.FwkReport.reportEvery = 1000
