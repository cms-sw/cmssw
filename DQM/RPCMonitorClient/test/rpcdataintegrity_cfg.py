import FWCore.ParameterSet.Config as cms

process = cms.Process("R2D")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.convention = 'Online'



############ 
process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")

process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")
############


# set maxevents; -1 -> take all
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2))

process.source = cms.Source ("PoolSource",fileNames =
cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RAW/v1/000/067/818/002ABA60-CDA4-DD11-9D53-001D09F248FD.root') )

# correct output file
process.MessageLogger = cms.Service("MessageLogger",
     debugModules = cms.untracked.vstring('rpcunpacker'),
     destinations = cms.untracked.vstring('cout'),
     cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)


process.p = cms.Path(process.rpcunpacker*process.rpcFEDIntegrity*process.dqmSaver)


