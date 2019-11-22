
import FWCore.ParameterSet.Config as cms

process = cms.Process("rpcDqmClient")

################# Input ########################
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2))

################# Geometry  ######################
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

################# DQM Cetral Modules ###################
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'RPC'
process.dqmSaver.convention = 'Online'


################# DQM Read ME ROOT File ####################
process.readMeFromFile = cms.EDAnalyzer("ReadMeFromFile",
      InputFile = cms.untracked.string('DQM_3.root')
)

################# DQM Event Summary ####################
process.load("DQM.RPCMonitorClient.RPCEventSummary_cfi")
process.rpcEventSummary.EventInfoPath = 'RPC/EventInfo'
process.rpcEventSummary.PrescaleFactor = 1

#process.rpcOccupancyTest = cms.EDAnalyzer("RPCOccupancyTest")

process.load("DQM.RPCMonitorClient.RPCDCSSummary_cfi")
process.load("DQM.RPCMonitorClient.RPCDaqInfo_cfi")
process.load("DQM.RPCMonitorClient.RPCDataCertification_cfi")

################# DQM Client Modules ####################
process.load("DQM.RPCMonitorClient.RPCDqmClient_cfi")
process.rpcdqmclient.RPCDqmClientList = cms.untracked.vstring("RPCNoisyStripTest","RPCOccupancyTest","RPCClusterSizeTest","RPCDeadChannelTest","RPCMultiplicityTest ")
process.rpcdqmclient.DiagnosticGlobalPrescale = cms.untracked.int32(5)
process.rpcdqmclient.NumberOfEndcapDisks  = cms.untracked.int32(3)

################### FED ##################################
process.load("DQM.RPCMonitorClient.RPCMonitorRaw_cfi")
process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")
process.load("DQM.RPCMonitorClient.RPCMonitorLinkSynchro_cfi")


 ################# Quality Tests #########################
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTesterRPC = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1)
)

################ Chamber Quality ##################
process.load("DQM.RPCMonitorClient.RPCChamberQuality_cfi")

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

################# Path ###########################
#process.rpcClientSequence = cms.Sequence(process.dqmEnv*process.readMeFromFile*process.qTesterRPC*process.rpcdqmclient*process.rpcOccupancyTest*process.rpcNoise*process.rpcChamberQuality*process.rpcEventSummary*process.dqmSaver)


process.p = cms.Path(process.readMeFromFile*process.dqmEnv*process.qTesterRPC*process.rpcdqmclient*process.rpcChamberQuality*process.rpcEventSummary*process.rpcDCSSummary*process.rpcDaqInfo*process.rpcDataCertification*process.dqmSaver)





