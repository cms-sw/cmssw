import FWCore.ParameterSet.Config as cms

process = cms.Process("R2D")
process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")

process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

process.load("DQM.RPCMonitorModule.rpcMonitorRaw_cfi")

process.load("DQMServices.Core.DQM_cfg")

# set maxevents; -1 -> take all
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(300))

process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring( 'file:r067647.root') )

# correct output file
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('rpcunpacker'),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)

process.rpcMonitorRaw.writeHistograms = True

process.p = cms.Path(process.rpcunpacker*process.rpcMonitorRaw)

