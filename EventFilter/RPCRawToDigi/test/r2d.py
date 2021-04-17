import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCR2D")

process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
process.RPCCabling.connect = 'sqlite_file:RPCEMap3.db'
process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

# set maxevents; -1 -> take all
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2))

#process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring( 'file:input.root'))
process.source = cms.Source ("NewEventStreamFileReader",fileNames = cms.untracked.vstring( 'file:input.dat'))


process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('rpcunpacker')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName =  cms.untracked.string('file:out.root'),
    outputCommands = cms.untracked.vstring("keep *")
)

process.p = cms.Path(process.rpcunpacker)
#process.ep = cms.EndPath(process.out)
