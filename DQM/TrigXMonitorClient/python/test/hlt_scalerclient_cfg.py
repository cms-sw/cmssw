# $Log: hlt_scalerclient_cfg.py,v $
# Revision 1.5  2008/09/03 02:13:47  wittich
# - bug fix in L1Scalers
# - configurable dqm directory in L1SCalers
# - other minor tweaks in HLTScalers
#
# Revision 1.4  2008/09/02 02:37:22  wittich
# - split L1 code from HLTScalers into L1Scalers
# - update cfi file accordingly
# - make sure to cd to correct directory before booking ME's
#
# Revision 1.3  2008/08/25 21:07:15  wittich
# updated py config file
#
# Revision 1.2  2008/08/24 16:34:56  wittich
# - rate calculation cleanups
# - fix error logging with LogDebug
# - report the actual lumi segment number that we think it might be
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")


### L1TGT unpacking
from DQM.L1TMonitor.L1TGT_unpack_cff import *
l1tgtpath = cms.Path(l1GtUnpack*l1GtEvmUnpack*cms.SequencePlaceholder("l1tgt"))

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




#process.load("DQM.L1TMonitor.L1TGT_unpack_cff")

#process.load("DQM.L1TMonitor.L1TGT_cfi")

process.load("DQM.TrigXMonitor.HLTScalers_cfi")
process.load("DQM.TrigXMonitorClient.HLTScalersClient_cfi")

process.load("DQM.TrigXMonitor.L1Scalers_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")


process.source = cms.Source("PoolSource",
#process.source = cms.Source("NewEventStreamFileReader",
    debugVerbosity = cms.untracked.uint32(1),
    debugVebosity = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring()
)
# from cff file
process.PoolSource.fileNames = cms.untracked.vstring(
#    'file:/tmp/wittich/0ECC6BE3-5F6F-DD11-A328-0019DB29C614.root',
#    'file:/tmp/wittich/26FD1D12-5E6F-DD11-8C6B-000423D6CA42.root',
#    'file:/tmp/wittich/3E414179-5E6F-DD11-B6DA-001617DBD472.root'
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/0ECC6BE3-5F6F-DD11-A328-0019DB29C614.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/26FD1D12-5E6F-DD11-8C6B-000423D6CA42.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/3E414179-5E6F-DD11-B6DA-001617DBD472.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/4E825423-5F6F-DD11-A891-001617C3B778.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/6C5C8A15-5D6F-DD11-9D27-000423D985E4.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/7A04421E-5F6F-DD11-8681-000423D992A4.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/7C3DC3B1-5F6F-DD11-8D31-001617C3B70E.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/8684AADF-5D6F-DD11-9082-001617DBD332.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/98F7037F-5D6F-DD11-88FE-000423D98DB4.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/A8C09AA5-5D6F-DD11-9DC8-000423D6B358.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/AC2B1871-5F6F-DD11-9949-001617C3B654.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/B499AEAB-5E6F-DD11-A84B-001617C3B6C6.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/CA095D4A-5E6F-DD11-A6DD-000423D6B48C.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/E884D9E4-5D6F-DD11-88AD-000423D6C8E6.root',
   '/store/data/Commissioning08/HLTDebug/RAW/CRUZET4_v1/000/058/042/EC646647-626F-DD11-A9AA-000423D98804.root'
)

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(2000)
    #input = cms.untracked.int32(200)
    input = cms.untracked.int32(-1)
)
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.MessageLogger = cms.Service("MessageLogger",
   #debugModules = cms.untracked.vstring('l1s', 'hlts', 'hltsClient', 'main_input'),
   debugModules = cms.untracked.vstring('hltsClient', 'hlts', 'main_input'),
   #debugModules = cms.untracked.vstring('*'),
                                    categories = cms.untracked.vstring('Status', 'Parameter'),
                                    noLineBreaks = cms.untracked.bool(True),
                                    destinations = cms.untracked.vstring('detailedInfo', 
                                                                         'critical', 
                                                                         'cout'),
                                    detailedInfo = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG'),
                                                                      DEBUG=cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                                                      Parameter=cms.untracked.PSet(limit=cms.untracked.int32(-1)),
                                                                      Status=cms.untracked.PSet(limit=cms.untracked.int32(-1)),
                                                                      Product=cms.untracked.PSet(limit=cms.untracked.int32(100)),
                                                                      FwkReport = cms.untracked.PSet(reportEvery = cms.untracked.int32(1000),
                                                                                                     limit = cms.untracked.int32(10000000)),
                                                                      ),
                                    critical = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'),
                                                              WARNING = cms.untracked.PSet(limit = cms.untracked.int32(0))),
                                    )

#process.MessageLogger.detailedInfo.FwkReport.reportEvery = 1000

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )
#process.assist = cms.Path(process.l1GtUnpack+process.l1tgt)

process.p = cms.EndPath(process.hlts+process.hltsClient)
process.p3 = cms.EndPath(process.l1s)

process.dqmWork = cms.Path(process.dqmEnv+process.dqmSaver)

# process.eca = cms.EDAnalyzer("EventContentAnalyzer")
# process.p4 = cms.Path(process.eca)

#process.hlts.verbose = True
process.hlts.l1GtData = cms.InputTag("hltGtDigis","","HLT")
process.l1s.l1GtData = cms.InputTag("hltGtDigis","","HLT")
#process.hlts.l1GtData = cms.InputTag("hltGtDigis","","")

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

## HLT REPORT
process.hltrep = cms.EDAnalyzer("HLTrigReport")
process.hltrep.HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
process.hltsum = cms.Path(process.hltrep)

