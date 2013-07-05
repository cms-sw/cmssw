import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMPathChecker")

process.load("DQMServices.Core.DQM_cfg")
#process.load("DQMOffline.Trigger.HLTGeneralOffline_cfi")


# Make this configuration inline
process.hltResults = cms.EDAnalyzer("GeneralHLTOffline",
    dirname = cms.untracked.string("HLT/General/paths"),
    muonRecoCollectionName = cms.untracked.string("muons"),
    plotAll = cms.untracked.bool(False),

    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    Nbins = cms.untracked.uint32(50),
    Nbins2D = cms.untracked.uint32(40),
    referenceBX= cms.untracked.uint32(1),
    NLuminositySegments= cms.untracked.uint32(2000),
    LuminositySegmentSize= cms.untracked.double(23),
    NbinsOneOverEt = cms.untracked.uint32(1000),

    muonEtaMax = cms.untracked.double(2.1),

    jetEtMin = cms.untracked.double(5.0),
    jetEtaMax = cms.untracked.double(3.0),

    electronEtMin = cms.untracked.double(5.0),

    photonEtMin = cms.untracked.double(5.0),

    tauEtMin = cms.untracked.double(10.0),
                          
     # this is I think MC and CRUZET4
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    HltProcessName = cms.string("HLTGRun"),
    processname = cms.string("HLT"),

    printWarnings = cms.untracked.bool(True)


 )


# Load up your offline processor
process.load("DQMServices.Components.DQMEnvironment_cfi")
#now you have a process.hltResults thing that will run

process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
#from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
#process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = 'auto:startup_GRun', )
#process.prefer("GlobalTag")
process.GlobalTag.globaltag = 'PRE_ST61_V1::All'




process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    # never use old files to do this test
    # you will be sad
#	 'file:/data/ndpc0/c/abrinke1/RAW/170354/SingleMu/08B0697A-B7B0-E011-B0DE-003048D375AA.root'
    'file:/data/ndpc0/c/slaunwhj/ONLINE/CMSSW_6_2_0_pre5/src/UserCode/slaunwhj/onlineDQMTest/onlineDQMContent/tttbar_620_pre5.root'
    
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

																						           
