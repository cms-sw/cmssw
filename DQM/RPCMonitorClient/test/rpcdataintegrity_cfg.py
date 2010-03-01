import FWCore.ParameterSet.Config as cms

process = cms.Process("rpcdqm")

################# Input ########################

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/733/DEB4AB86-F0C3-DE11-8BB8-000423D6BA18.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/683/783CDCC2-FDC3-DE11-8534-003048D2C108.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/664/68297EC3-F1C3-DE11-B124-0019B9F730D2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/651/EAF2245E-E2C3-DE11-9399-001D09F295FB.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/651/DC14ABB3-E3C3-DE11-8F35-0030487A1FEC.root'] );

secFiles.extend( [
               ] )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

################# Geometry  ######################
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

################# RPC Unpacker  ######################
process.rpcunpacker = cms.EDProducer("RPCUnpackingModule",
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



#process.rpcMultiplicityTest = cms.EDAnalyzer("RPCMultiplicityTest")
process.rpcOccupancyTest = cms.EDAnalyzer("RPCOccupancyChipTest")
process.rpcClusterSize = cms.EDAnalyzer("RPCClusterSizeTest")
process.load("DQM.RPCMonitorClient.RPCMon_SS_Dbx_Global_cfi")

################# Quality Tests #########################
process.qTesterRPC = cms.EDAnalyzer("QualityTester",
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


################# DQM Read ME ROOT File ####################
process.readME = cms.EDAnalyzer("ReadMeFromFile",
      InputFile = cms.untracked.string('DQM_V0_MERGED_R70664.root')
)

process.rpcClientModule = cms.EDAnalyzer("RPCDqmClient")

################# Quality module ########################
process.rpcquality = cms.EDAnalyzer("RPCChamberQuality")



################# Path ###########################
process.rpcDigi = cms.Sequence(process.rpcdigidqm)
process.rpcClient = cms.Sequence(process.qTesterRPC*process.rpcClientModule*process.rpcClusterSize*process.rpcOccupancyTest*process.dqmEnv*process.rpcquality*process.dqmSaver)

process.p = cms.Path(process.rpcClient)


