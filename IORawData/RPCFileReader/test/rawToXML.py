import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCR2X")

# set maxevents; -1 -> take all
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))

process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring( 'file:input.root'))

# correct output file
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('r2x'),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG'))
)

process.r2x= cms.EDFilter("RawToXML",
    InputLabel = cms.InputTag("source"),
    xmlFileName = cms.string("myEvent.xml")
)

process.p = cms.Path(process.r2x)



