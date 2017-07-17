import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

#-----------------------------
#  DQM SOURCES
#-----------------------------

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = 'CRZT210_V1H::All'
#process.prefer("GlobalTag")




process.load("DQM.L1TMonitor.L1TGT_unpack_cff")

process.load("DQM.L1TMonitor.L1TGT_cfi")

process.load("DQM.TrigXMonitor.HLTScalers_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.TrigXMonitor.run58738")

# process.source = cms.Source("PoolSource",
# #process.source = cms.Source("NewEventStreamFileReader",
#     debugVerbosity = cms.untracked.uint32(1),
#     debugVebosity = cms.untracked.bool(True),
#     fileNames = cms.untracked.vstring(
# 	#'file:/cms/mon/data/lookarea_SM/GlobalCruzet3.00051218.0001.DQM.storageManager.0.0000.dat'
# 	'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root',
#         'file:/tmp/wteo/AC5529A1-D754-DD11-9362-000423D9863C.root',
#         'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root',
#         'file:/tmp/wteo/AC5529A1-D754-DD11-9362-000423D9863C.root',
#         'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root',
#         'file:/tmp/wteo/AC5529A1-D754-DD11-9362-000423D9863C.root',
#         'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root',
#         'file:/tmp/wteo/AC5529A1-D754-DD11-9362-000423D9863C.root'
#         #'file:/cms/mon/data/lookarea_SM/GlobalCruzet3MW33.00056489.0001.HLTDEBUG.storageManager.2.0000.dat'
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root', 
#         #'file:/tmp/wteo/28E1D7F9-214C-DD11-B42C-000423D9880C.root'
#         )
# )

# process.maxEvents = cms.untracked.PSet(
#     #input = cms.untracked.int32(1000)
#     #input = cms.untracked.int32(10)
#     input = cms.untracked.int32(-1)
# )
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

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

process.p = cms.EndPath(process.hlts)

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


