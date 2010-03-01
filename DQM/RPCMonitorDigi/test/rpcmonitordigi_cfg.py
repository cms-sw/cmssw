
import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCDQM")

############# Source File ########################
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RECO/v1/000/070/664/1CE1633D-87AF-DD11-AD95-000423D98B08.root')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(15000))


################ Condition ######################
process.load("CondCore.DBCommon.CondDBSetup_cfi")


################ DQM #######################
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'RPC'
process.dqmSaver.convention = 'Online'


############ Geometry ################################
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")



process.load("Configuration.StandardSequences.MagneticField_cff")

##### RAW to DIGI #####
process.rpcunpacker = cms.EDProducer("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("source"),
    doSynchro = cms.bool(False)
)

########## RecHits ##########################
process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")
process.rpcRecHits.rpcDigiLabel = 'muonRPCDigis'
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")



###################### RPC Monitor Digi ###################
process.load("DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi")
process.rpcdigidqm.DigiEventsInterval = 100
process.rpcdigidqm.DigiDQMSaveRootFile = True
process.rpcdigidqm.dqmshifter = True
process.rpcdigidqm.dqmexpert = True
process.rpcdigidqm.dqmsuperexpert = True
process.rpcdigidqm.RootFileNameDigi = 'DQM_3.root'

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
process.rpcdqmclient.RPCDqmClientList = cms.untracked.vstring("RPCNoisyStripTest","RPCOccupancyTest","RPCClusterSizeTest","RPCDeadChannelTest","RPCMultiplicityTest")
#process.rpcdqmclient.DiagnosticGlobalPrescale = cms.untracked.int32(5)
process.rpcdqmclient.NumberOfEndcapDisks  = cms.untracked.int32(3)

################### FED ##################################
process.load("DQM.RPCMonitorClient.RPCMonitorRaw_cfi")
process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")
process.load("DQM.RPCMonitorClient.RPCMonitorLinkSynchro_cfi")

################# Quality Tests #########################
process.qTesterRPC = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1)
)

################ Chamber Quality ##################
process.load("DQM.RPCMonitorClient.RPCChamberQuality_cfi")

############### Output Module ######################
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('RPCDQM.root')
)

                               
############# Path ########################

process.p = cms.Path(process.rpcRecHits*process.rpcdigidqm*process.dqmEnv*process.qTesterRPC*process.rpcdqmclient*process.rpcChamberQuality*process.rpcEventSummary*process.rpcDCSSummary*process.rpcDaqInfo*process.rpcDataCertification*process.dqmSaver)



