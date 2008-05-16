import FWCore.ParameterSet.Config as cms

process = cms.Process("HEXDISPLAY")
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    #untracked vstring fileNames = {'file:/data/scooper/data/grea07/40792E58-B757-DC11-8AB2-001617E30F46.root'}
    fileNames = cms.untracked.vstring('file:/data/scooper/data/gren07/P5_Co.00029485.A.0.0.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(150)
)
process.hexDump = cms.EDFilter("EcalHexDisplay",
    verbosity = cms.untracked.int32(0),
    filename = cms.untracked.string('dump.bin'),
    # fed_id: EE- is 601-609,  EB is 610-645,  EE- is 646-654
    # when using 'single sm' fed corresponds to construction number  
    beg_fed_id = cms.untracked.int32(0),
    writeDCC = cms.untracked.bool(False),
    end_fed_id = cms.untracked.int32(654)
)

process.counter = cms.OutputModule("AsciiOutputModule")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.hexDump)
process.ep = cms.EndPath(process.counter)

