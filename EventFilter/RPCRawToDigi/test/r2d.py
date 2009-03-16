import FWCore.ParameterSet.Config as cms

process = cms.Process("R2D")
process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")

process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

process.load("DQMServices.Core.DQM_cfg")

# set maxevents; -1 -> take all
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(300))

process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(
'/store/data/Commissioning08/Cosmics/RAW/v1/000/070/036/86276C9B-72AD-DD11-90F9-000423D6C8EE.root'
))

# correct output file
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('rpcunpacker'),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName =  cms.untracked.string('file:rawdata.root'),
    outputCommands = cms.untracked.vstring("keep *")
)

process.p = cms.Path(process.rpcunpacker)
#process.ep = cms.EndPath(process.out)
