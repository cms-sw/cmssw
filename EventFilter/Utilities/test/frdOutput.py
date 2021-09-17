import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os
import math


options = VarParsing.VarParsing ('analysis')

options.register ('runNumber',
                  100, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('eventsPerLS',
                  105,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Max LS to generate (0 to disable limit)")

options.register ('fedMeanSize',
                  1024,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Mean size of generated (fake) FED raw payload")

options.register ('frdFileVersion',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Generate raw files with FRD file header with version 1 or separate JSON files with 0")



options.parseArguments()

process = cms.Process("RRDOUTPUT")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.eventsPerLS)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" )),
    destinations = cms.untracked.vstring( 'cout' )
)

process.source = cms.Source("EmptySource",
     firstRun= cms.untracked.uint32(options.runNumber),
     numberEventsInLuminosityBlock = cms.untracked.uint32(options.eventsPerLS),
     numberEventsInRun       = cms.untracked.uint32(0)
)

process.a = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(0))

process.s = cms.EDProducer("DaqFakeReader",
                           meanSize = cms.untracked.uint32(options.fedMeanSize),
                           width = cms.untracked.uint32(int(math.ceil(options.fedMeanSize/2.))),
                           tcdsFEDID = cms.untracked.uint32(1024),
                           injectErrPpm = cms.untracked.uint32(0)
                           )

process.out = cms.OutputModule("FRDOutputModule",
    source = cms.InputTag("s"),
    frdVersion = cms.untracked.uint32(6),
    frdFileVersion = cms.untracked.uint32(options.frdFileVersion),
#    fileName = cms.untracked.string("frd_output.raw")
    )

process.p = cms.Path(process.s+process.a)

process.ep = cms.EndPath(process.out)
