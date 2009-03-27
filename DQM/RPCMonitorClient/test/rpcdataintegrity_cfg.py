import FWCore.ParameterSet.Config as cms

process = cms.Process("rpcdqm")

################# Input ########################
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

################# Geometry  ######################
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

################# RPC Unpacker  ######################
process.rpcunpacker = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("source"),
    doSynchro = cms.bool(False)
)

process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")

################# RPC Rec Hits  ######################
process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")
process.rpcRecHits.rpcDigiLabel = 'rpcunpacker'

################# DQM Cetral Modules ###################
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'RPC'
process.dqmSaver.convention = 'Online'
################# DQM Digi Module ######################
process.load("DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi")
process.rpcdigidqm.DigiEventsInterval = 100
process.rpcdigidqm.dqmshifter = True
process.rpcdigidqm.dqmexpert = True
process.rpcdigidqm.dqmsuperexpert = False
process.rpcdigidqm.DigiDQMSaveRootFile = False

################# DQM Client Modules ####################
process.load("DQM.RPCMonitorClient.RPCEventSummary_cfi")
process.rpcEventSummary.PrescaleFactor = 1



<<<<<<< rpcdataintegrity_cfg.py
=======
process.rpcOccupancyTest = cms.EDAnalyzer("RPCOccupancyChipTest")
>>>>>>> 1.7

#process.rpcMultiplicityTest = cms.EDAnalyzer("RPCMultiplicityTest")
process.rpcOccupancyTest = cms.EDAnalyzer("RPCOccupancyChipTest")
process.rpcClusterSize = cms.EDAnalyzer("RPCClusterSizeTest")
process.load("DQM.RPCMonitorClient.RPCMon_SS_Dbx_Global_cfi")

################# Quality Tests #########################
process.qTesterRPC = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1)
)

############ FED Integrity ############################
process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")

############# Message Logger ####################
process.MessageLogger = cms.Service("MessageLogger",
     debugModules = cms.untracked.vstring('rpcunpacker'),
     destinations = cms.untracked.vstring('cout'),
     cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)

<<<<<<< rpcdataintegrity_cfg.py

################# DQM Read ME ROOT File ####################
process.readME = cms.EDAnalyzer("ReadMeFromFile",
      InputFile = cms.untracked.string('DQM_V0_MERGED_R70664.root')
)

process.rpcClientModule = cms.EDAnalyzer("RPCDqmClient")

################# Quality module ########################
process.rpcquality = cms.EDAnalyzer("RPCChamberQuality")



=======

################# DQM Read ME ROOT File ####################
process.readME = cms.EDAnalyzer("ReadMeFromFile",
      InputFile = cms.untracked.string('DQM_V0_MERGED_R70664.root')
)


>>>>>>> 1.7
################# Path ###########################
process.rpcDigi = cms.Sequence(process.rpcunpacker*process.rpcRecHits*process.rpcdigidqm*process.rpcAfterPulse)
<<<<<<< rpcdataintegrity_cfg.py
process.rpcClient = cms.Sequence(process.readME*process.qTesterRPC*process.rpcClientModule*process.rpcClusterSize*process.rpcOccupancyTest*process.dqmEnv*process.rpcquality*process.dqmSaver)

process.p = cms.Path(process.rpcClient)
=======
process.rpcClient = cms.Sequence(process.readME*process.qTesterRPC*process.RPCDeadChannelTest*process.rpcOccupancyTest*process.dqmEnv*process.rpcEventSummary*process.dqmSaver)

process.p = cms.Path(process.rpcClient)
>>>>>>> 1.7


