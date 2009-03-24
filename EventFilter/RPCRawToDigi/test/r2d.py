import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCR2D")

process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

# set maxevents; -1 -> take all
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(
#'file:/tmp/konec/run070036.root'
'/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/6C75241D-3A09-DE11-BE1B-001D09F29597.root'
#'/store/data/Commissioning08/Cosmics/RAW/v1/000/070/036/86276C9B-72AD-DD11-90F9-000423D6C8EE.root'
))

# correct output file
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('rpcunpacker'),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG'))
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName =  cms.untracked.string('file:out.root'),
    outputCommands = cms.untracked.vstring("keep *")
)

process.p = cms.Path(process.rpcunpacker)
#process.ep = cms.EndPath(process.out)
