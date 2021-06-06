import FWCore.ParameterSet.Config as cms

process = cms.Process("rpcDqmClient")


## InputFile = DQM root file path
process.readMeFromFile = cms.EDAnalyzer("ReadMeFromFile",
      InputFile = cms.untracked.string('/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/Express/121/964/DQM_V0001_R000121964__StreamExpress__BeamCommissioning09-Express-v2__DQM.root'),                                  
)

####################################### DO NOT CHANGE  #############################################
process.load("DQM.RPCMonitorClient.RPCDqmClient_cfi")
process.rpcdqmclient.RPCDqmClientList = cms.untracked.vstring("RPCBxTest")
###################################################################################################


## RMSCut = maximum RMS allowed
## EntriesCut = minimum entries allowed
## DistanceFromZeroBx = maximum distance from BX 0 in absolute value (Rolls that will be written in file)  
process.rpcdqmclient.RMSCut = cms.untracked.double(1.1)
process.rpcdqmclient.EntriesCut = cms.untracked.int32(10)
process.rpcdqmclient.DistanceFromZeroBx = cms.untracked.double(1.5)



####################################### DO NOT CHANGE  #############################################
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1)) 

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")


process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'RPC'
process.dqmSaver.convention = 'Online'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('rpcbxtest')
)

process.p = cms.Path(process.readMeFromFile*process.rpcdqmclient*process.dqmEnv*process.dqmSaver)
####################################################################################################





