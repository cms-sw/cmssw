import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')

options.register ('compAlgo',
                  'ZLIB', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Compression Algorithm")

options.parseArguments()


process = cms.Process("HLT")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

process.source = cms.Source("EmptySource",
    firstEvent = cms.untracked.uint64(10123456789)
)

process.m1 = cms.EDProducer("StreamThingProducer",
    instance_count = cms.int32(5),
    array_size = cms.int32(2)
)

process.m2 = cms.EDProducer("NonProducer")

process.a1 = cms.EDAnalyzer("StreamThingAnalyzer",
    product_to_get = cms.string('m1')
)

process.out = cms.OutputModule("EventStreamFileWriter",
    fileName = cms.untracked.string('teststreamfile.dat'),
    padding = cms.untracked.uint32(4096),
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    compression_algorithm = cms.untracked.string(options.compAlgo),
    max_event_size = cms.untracked.int32(7000000)
)

process.p1 = cms.Path(process.m1*process.a1*process.m2)
process.end = cms.EndPath(process.out)
