
import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCDQM")

############# Source File ########################

process.source = cms.Source("PoolSource",
#      fileNames = cms.untracked.vstring('/store/data/Commissioning10/Cosmics/RAW/v1/000/125/838/702BD989-F60B-DF11-A49C-0030487CD77E.root')
     fileNames = cms.untracked.vstring('/store/data/Commissioning10/Cosmics/RECO/v1/000/125/838/C6D4F60E-FA0B-DF11-BA84-003048D37560.root')
#      fileNames = cms.untracked.vstring('/store/data/Commissioning10/RandomTriggers/RAW/v3/000/128/736/0C1ED6D3-311E-DF11-B20E-000423D99EEE.root')

)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))


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

## process.rpcunpacker = cms.EDFilter("RPCUnpackingModule",
##     InputLabel = cms.InputTag("source"),
##     doSynchro = cms.bool(False)
## )
process.load("EventFilter.RPCRawToDigi.RPCFrontierCabling_cfi")


########## RecHits ##########################
process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")
#process.rpcRecHits.rpcDigiLabel ='rpcunpacker'
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

################# After Pulse ####################
process.load("DQM.RPCMonitorClient.RPCMon_SS_Dbx_Global_cfi")


################# DQM Client Modules ####################
process.load("DQM.RPCMonitorClient.RPCDqmClient_cfi")
process.rpcdqmclient.RPCDqmClientList = cms.untracked.vstring("RPCNoisyStripTest","RPCOccupancyTest","RPCClusterSizeTest","RPCDeadChannelTest","RPCMultiplicityTest")
process.rpcdqmclient.DiagnosticGlobalPrescale = cms.untracked.int32(1)
process.rpcdqmclient.NumberOfEndcapDisks  = cms.untracked.int32(3)


################### FED ##################################
process.load("DQM.RPCMonitorClient.RPCMonitorRaw_cfi")
process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")
process.rpcFEDIntegrity.RPCRawCountsInputTag = 'provaDiNoCrash'
process.load("DQM.RPCMonitorClient.RPCMonitorLinkSynchro_cfi")

################# Quality Tests #########################
## process.qTesterRPC = cms.EDFilter("QualityTester",
##     qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
##     prescaleFactor = cms.untracked.int32(1)
## )

################ Chamber Quality ##################
process.load("DQM.RPCMonitorClient.RPCChamberQuality_cfi")

############### Output Module ######################
process.out = cms.OutputModule("PoolOutputModule",
   fileName = cms.untracked.string('RPCDQM.root'),
   outputCommands = cms.untracked.vstring("keep *")
)



############# Message Logger ####################
process.MessageLogger = cms.Service("MessageLogger",
     debugModules = cms.untracked.vstring('*'),
     cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
     destinations = cms.untracked.vstring('cout')
 )




################ Memory check ##################
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
 #       ignoreTotal = cms.untracked.int32(1) ## default is one
#) 

#################Timing ###############
#process.Timing = cms.Service("Timing")
#process.options = cms.untracked.PSet(
 #        wantSummary = cms.untracked.bool(True)
#)


#process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))
#process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True))


############# Path ########################


#process.p = cms.Path(process.rpcRecHits*process.rpcdigidqm*process.dqmEnv*process.qTesterRPC*process.rpcdqmclient*process.rpcChamberQuality*process.dqmSaver)

process.p = cms.Path(process.rpcRecHits*process.rpcdigidqm*process.rpcAfterPulse*process.rpcFEDIntegrity*process.dqmEnv*process.qTesterRPC*process.rpcdqmclient*process.rpcChamberQuality*process.rpcEventSummary*process.rpcDCSSummary*process.rpcDaqInfo*process.rpcDataCertification*process.dqmSaver)

process.e = cms.EndPath(process.out)


