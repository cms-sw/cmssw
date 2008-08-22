import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

#-----------------------------
#  DQM SOURCES
#-----------------------------

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = 'CRZT210_V1H::All'
#process.prefer("GlobalTag")




process.load("DQM.L1TMonitor.L1TGT_unpack_cff")

process.load("DQM.L1TMonitor.L1TGT_cfi")

process.load("DQM.TrigXMonitor.HLTScalers_cfi")
process.load("DQM.TrigXMonitorClient.HLTScalersClient_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQM.TrigXMonitorClient.run58042_cfi")
process.load("DQM.TrigXMonitorClient.run58042_cfi")


process.source = cms.Source("PoolSource",
#process.source = cms.Source("NewEventStreamFileReader",
    debugVerbosity = cms.untracked.uint32(1),
    debugVebosity = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring()
)
# from cff file
process.PoolSource.fileNames = inputFileNames

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(1000)
    #input = cms.untracked.int32(10)
    input = cms.untracked.int32(-1)
)
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.assist = cms.Path(process.l1GtUnpack+process.l1tgt)

process.p = cms.EndPath(process.hlts+process.hltsClient)

process.dqmWork = cms.Path(process.dqmEnv+process.dqmSaver)

#process.hlts.verbose = True
process.hlts.l1GtData = cms.InputTag("l1GtUnpack","","DQM")

###########################
###   DQM Environment   ###
###########################

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'srv-c2d05-12'
process.DQM.collectorPort = 9190
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.saveAtJobEnd = False


