import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing()
options.register("maxEvents", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "max events to process")
options.register("output", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "output filename")
options.register("intprod", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "int value to produce")
options.parseArguments()

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")

process.maxEvents.input = options.maxEvents

process.m1a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(options.intprod)
)
process.p1 = cms.Path(process.m1a)

if len(options.output)>0:
    process.testout1 = cms.OutputModule("TestOutputModule",
        name = cms.string(options.output),
    )
    process.e1 = cms.EndPath(process.testout1)
