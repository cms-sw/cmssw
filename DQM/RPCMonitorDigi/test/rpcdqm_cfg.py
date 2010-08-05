
import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCDQM")

############# Source File ########################

process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring('/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/637/8C53F5F6-4062-DF11-BFE3-00304879FA4C.root',
                                       '/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/637/AAF9112F-3C62-DF11-9C54-0030487CD16E.root'
                                       )
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

################ Condition #######################
process.load("CondCore.DBCommon.CondDBSetup_cfi")

############ Geometry ############################
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

################### DQM ##########################
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'RPC'
process.dqmSaver.convention = 'Online'

############# RPC Monitor Digi ###################
process.load("DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi")
process.rpcdigidqm.SaveRootFile = False
process.rpcdigidqm.RootFileNameDigi = 'DQM_3.root'

############### DQM Client Modules ###############
process.load("DQM.RPCMonitorClient.RPCDqmClient_cfi")
process.rpcnoiseclient.MinimumRPCEvents = 1
process.rpcmuonclient.MinimumRPCEvents = 1
process.rpcnoiseclient.DiagnosticPrescale = 1
process.rpcmuonclient.DiagnosticPrescale = 1

################# Quality Tests ##################
process.qTesterRPC = cms.EDAnalyzer("QualityTester",
                                  qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
                                  prescaleFactor = cms.untracked.int32(5)
                                  )

################ Chamber Quality #################
process.load("DQM.RPCMonitorClient.RPCChamberQuality_cfi")


############## Message Logger ####################
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
                                    )

############### Output Module ####################
process.out = cms.OutputModule("PoolOutputModule",
   fileName = cms.untracked.string('RPCDQM.root'),
   outputCommands = cms.untracked.vstring("keep *")
)

#################### Path ########################
process.rpcdqmsource = cms.Sequence(process.rpcdigidqm)
process.rpcdqmclient = cms.Sequence(process.qTesterRPC * process.rpcnoiseclient * process.rpcmuonclient * process.dqmSaver)
process.p = cms.Path(process.rpcdqmsource*process.rpcdqmclient)




#process.e = cms.EndPath(process.out)
