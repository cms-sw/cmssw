import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

#-----------------------------
#  DQM SOURCES
#----------------------------- 

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_31X_V1P::All"
process.prefer("GlobalTag")


process.load("DQM.L1TMonitor.L1TMonitor_cff")
process.load("DQM.L1TMonitorClient.L1TMonitorClient_cff")
process.load("DQM.TrigXMonitor.L1Scalers_cfi")
process.load("DQM.TrigXMonitorClient.L1TScalersClient_cfi")
process.l1s.l1GtData = cms.InputTag("l1GtUnpack","","DQM")
process.l1s.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM") 
process.l1tsClient.dqmFolder = cms.untracked.string("L1T/L1Scalers_SM")
process.p3 = cms.EndPath(process.l1s+process.l1tsClient)


##  Available data masks (case insensitive):
##    all, gt, muons, jets, taujets, isoem, nonisoem, met
process.l1tEventInfoClient.dataMaskedSystems =cms.untracked.vstring("Jets","TauJets","IsoEm","NonIsoEm","MET")

##  Available emulator masks (case insensitive):
##    all, dttf, dttpg, csctf, csctpg, rpc, gmt, ecal, hcal, rct, gct, glt
process.l1tEventInfoClient.emulatorMaskedSystems = cms.untracked.vstring("All")

#process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(3000)
#)

process.load("DQM.L1TMonitor.119580_cfi")

#process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('file:/cms/mon/data/lookarea_SM/GlobalCruzet3.00051488.0001.DQM.storageManager.0.0000.dat')
    #fileNames = cms.untracked.vstring('file:/cms/mon/data/lookarea_SM/GlobalCruzet3.00051437.0001.DQM.storageManager.1.0000.dat')
    #fileNames = cms.untracked.vstring('file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root')
#    fileNames = cms.untracked.vstring('/store/data/Commissioning09/Cosmics/RAW/v3/000/119/400/18FA115E-D9C8-DE11-B7AA-001D09F24399.root')
    #fileNames = cms.untracked.vstring('file:/cms/mon/data/lookarea_SM/GlobalCruzet3.00051218.0001.DQM.storageManager.0.0000.dat')
    #fileNames = cms.untracked.vstring('file:/tmp/wteo/E244612F-7751-DD11-8931-000423D94700.root')
    #fileNames = cms.untracked.vstring('file:/tmp/wteo/001365AC-1C1C-DD11-AA0B-0030487D62E6.root')
    #fileNames = cms.untracked.vstring('file:/tmp/wteo/1CD42767-9B60-DD11-B56E-001617DBD224.root')


#)

#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring(
#                                      '/store/streamer/RunPrep09/A/000/119/580/RunPrep09.00119580.0305.A.storageManager.02.0000.dat'
#				      )
#)				      

process.DQMStore.verbose = 0
process.DQM.collectorHost = "srv-c2d05-12"
process.DQM.collectorPort = 9190
#process.DQMStore.referenceFileName = "DQM_L1T_R000002467.root"
#process.DQMStore.referenceFileName = "/nfshome0/wteo/test/rctL1tDqm.root"


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

process.DQMStore.verbose = 0
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'L1T'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

