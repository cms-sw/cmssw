import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing()
options.register("maxEvents", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "max events to process")
options.register("threads", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "number of threads")
options.parseArguments()

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")

process.maxEvents.input = options.maxEvents
process.options.numberOfThreads = options.threads
