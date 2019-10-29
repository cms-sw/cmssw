import FWCore.ParameterSet.Config as cms

process = cms.Process("rpcdqm")

################# Input ########################

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

secFiles.extend([]);

readFiles.extend( ['/store/data/Commissioning10/Cosmics/RAW/v4/000/132/202/0224D729-6A38-DF11-B1AF-0030487C8CBE.root'] );


################# Geometry  ######################
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

################# RPC Unpacker  ######################
process.rpcunpacker = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.InputTag("source"),
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

################# DQM Event Summary ####################
process.load("DQM.RPCMonitorClient.RPCEventSummary_cfi")
process.rpcEventSummary.EventInfoPath = 'RPC/EventInfo'
process.rpcEventSummary.PrescaleFactor = 1

process.load("DQM.RPCMonitorClient.RPCDCSSummary_cfi")
process.load("DQM.RPCMonitorClient.RPCDaqInfo_cfi")
process.load("DQM.RPCMonitorClient.RPCDataCertification_cfi")

################# DQM Digi Module ######################
process.load("DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi")
process.rpcdigidqm.DigiEventsInterval = 10
process.rpcdigidqm.dqmshifter = True
process.rpcdigidqm.dqmexpert = True
process.rpcdigidqm.dqmsuperexpert = False
process.rpcdigidqm.DigiDQMSaveRootFile = False

################# DQM Client Modules ####################
process.load("DQM.RPCMonitorClient.RPCEventSummary_cfi")
process.rpcEventSummary.PrescaleFactor = 1

process.load("DQM.RPCMonitorClient.RPCDqmClient_cfi")
process.rpcdqmclient.RPCDqmClientList = cms.untracked.vstring("RPCNoisyStripTest","RPCOccupancyTest","RPCClusterSizeTest","RPCDeadChannelTest","RPCMultiplicityTest")
process.rpcdqmclient.DiagnosticGlobalPrescale = cms.untracked.int32(1)
process.rpcdqmclient.NumberOfEndcapDisks  = cms.untracked.int32(3)
process.rpcdqmclient.MinimumRPCEvents = cms.untracked.int32(1)

process.load("DQM.RPCMonitorClient.RPCMon_SS_Dbx_Global_cfi")

################### FED ##################################
process.load("DQM.RPCMonitorClient.RPCMonitorRaw_cfi")
process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")
process.load("DQM.RPCMonitorClient.RPCMonitorLinkSynchro_cfi")


 ################# Quality Tests #########################
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTesterRPC = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1),
    qtestOnEndLumi =  cms.untracked.bool(True)                               
)

################ Chamber Quality ##################
process.load("DQM.RPCMonitorClient.RPCChamberQuality_cfi")
process.rpcChamberQuality.minEvents =  cms.untracked.int32(1)

############# Message Logger ####################
process.MessageLogger = cms.Service("MessageLogger",
     debugModules = cms.untracked.vstring('rpcdqmclient'),
     destinations = cms.untracked.vstring('cout'),
     cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)


#process.Timing = cms.Service('Timing')

## process.options = cms.untracked.PSet(
##          wantSummary = cms.untracked.bool(True)
##          )


############## Output module ##################*_MEtoEDMConverter_*_*
process.out = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('out.root'),
     outputCommands = cms.untracked.vstring("keep *")
)


################# Path ###########################
#process.rpcClientSequence = cms.Sequence(process.dqmEnv*process.readMeFromFile*process.qTesterRPC*process.rpcdqmclient*process.rpcOccupancyTest*process.rpcNoise*process.rpcChamberQuality*process.rpcEventSummary*process.dqmSaver)


process.p = cms.Path(process.rpcunpacker*process.rpcRecHits*process.rpcdigidqm*process.rpcAfterPulse*process.rpcMonitorRaw*process.dqmEnv*process.qTesterRPC*process.rpcdqmclient*process.rpcChamberQuality*process.rpcEventSummary*process.rpcDCSSummary*process.rpcDaqInfo*process.rpcDataCertification*process.rpcFEDIntegrity*process.dqmSaver)


#process.p = cms.Path(process.rpcunpacker*process.rpcRecHits*process.rpcdigidqm*process.dqmSaver)
