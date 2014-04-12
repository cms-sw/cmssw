import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMPathChecker")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMOffline.Trigger.HLTGeneralOffline_cfi")

# Load up your offline processor
process.load("DQMServices.Components.DQMEnvironment_cfi")
#now you have a process.hltResults thing that will run


process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'GR_R_52_V9::All'
process.prefer("GlobalTag")



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    # never use old files to do this test
    # you will be sad
#	 'file:/data/ndpc0/c/abrinke1/RAW/170354/SingleMu/08B0697A-B7B0-E011-B0DE-003048D375AA.root'
    '/store/relval/CMSSW_5_2_7/RelValTTbar/GEN-SIM-RECO/PU_START52_V10-v1/0004/12EAF8E1-3708-E211-8649-003048D2C108.root'
    
    )
)


#process.source = cms.Source("NewEventStreamFileReader",
#							fileNames = cms.untracked.vstring('file:/data/ndpc1/b/slaunwhj/ONLINE/CMSSW_3_5_0_patch1/src/Analysis/onlineDQMTest/Data.00127708.0001.DQM.storageManager.00.0000.dat')
#
#    fileNames = cms.untracked.vstring('file:/nfshome0/lorenzo/305FBD08-D78F-DD11-A1D6-001617C3B65A.root')
#							
 #   fileNames = cms.untracked.vstring(
#         'file:/cmsdisk1/lookarea_SM/GlobalMW40.00064245.0001.HLTDEBUG.storageManager.01.0000.dat',
#         'file:/cmsdisk1/lookarea_SM/GlobalMW40.00064210.0001.HLTDEBUG.storageManager.03.0000.dat'
#   )
#)


#process.load("DQM.TrigXMonitor.HLTScalers_cfi")
#process.load("DQM.TrigXMonitorClient.HLTScalersClient_cfi")
# Remove this because we don't care about L1
#process.load("DQM.TrigXMonitor.HLTSeedL1LogicScalers_cfi")
#process.hlts.l1GtData = cms.InputTag("l1GtUnpack","","DQM")
#process.hlts.dqmFolder = cms.untracked.string("HLT/HLTScalers_SM")
#process.hltsClient.dqmFolder = cms.untracked.string("HLT/HLTScalers_SM")
#process.hltSeedL1Logic.l1GtData = cms.InputTag("l1GtUnpack","","DQM")
#process.hltSeedL1Logic.dqmFolder =    cms.untracked.string("HLT/HLTSeedL1LogicScalers_SM")




process.p = cms.EndPath(process.hltResults)
  
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
# #process.MessageLogger.categories = ['hltResults']
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



outfile = open('config.py','w')
print >> outfile,process.dumpPython()
outfile.close()

																						           
