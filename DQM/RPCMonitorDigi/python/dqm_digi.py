####### By David Lomidze ##########

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.load("EventFilter.RPCRawToDigi.RPCUnpacking_cfi")

process.load("DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

#process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

##### RAW to DIGI #####
process.rpcunpacker = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("source"),
    doSynchro = cms.bool(False)
)


process.source = cms.Source("PoolSource",
    moduleLogName = cms.untracked.string('source'),
 #   fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RECO/v1/000/070/659/50CD2EE7-79AF-DD11-918C-000423D9870C.root ')
     fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RECO/v1/000/070/664/1CE1633D-87AF-DD11-AD95-000423D98B08.root')
     # fileNames = cms.untracked.vstring('file:/tmp/dlomidze/digi.root ')                       
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('RPCDQM.root')
)

process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'RPC'


################# DQM Client Modules ####################
process.load("DQM.RPCMonitorClient.RPCEventSummary_cfi")
process.rpcEventSummary.EventInfoPath = 'RPC/EventInfo'
process.rpcEventSummary.RPCPrefixDir = 'RPC/RecHits'
process.rpcEventSummary.RPCPrefixDir = 'RPC/RecHits'
process.rpcEventSummary.PrescaleFactor = 10
process.load("DQM.RPCMonitorClient.RPCMon_SS_Dbx_Global_cfi")

################# Quality Tests #########################
process.qTesterRPC = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1)
)
                               
#prosess.rpcfedIntegrity = cms.RPCFEDIntegrity("RPCFEDIntegrity")
#
process.RPCDeadChannelTest = cms.EDAnalyzer("RPCDeadChannelTest")
#process.RPCOccupancyTest = cms.EDAnalyzer("RPCOccupancyTest")

process.p = cms.Path(process.rpcRecHits*process.rpcdigidqm*process.dqmEnv)
process.rpcdigidqm.DigiEventsInterval = 100
process.rpcdigidqm.DigiDQMSaveRootFile = True
process.rpcdigidqm.dqmshifter = True
process.rpcdigidqm.dqmexpert = True
process.rpcdigidqm.dqmsuperexpert = True
process.rpcdigidqm.RootFileNameDigi = 'DQM_3.root'
process.DQM.collectorHost = ''
#process.DQM.collectorPort = 9090
#process.DQM.debug = False
process.rpcRecHits.rpcDigiLabel = 'muonRPCDigis'


