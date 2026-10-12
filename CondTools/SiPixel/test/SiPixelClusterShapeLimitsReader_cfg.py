import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# Reads a SiPixelClusterShapeLimits payload from a sqlite file and dumps its content

options = VarParsing.VarParsing()
options.register('inputDB', 'SiPixelClusterShapeLimits.db', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 'input sqlite file')
options.register('inputTag', 'SiPixelClusterShapeLimits_phase1_v1', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 'input tag')
options.register('run', 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'run number to read')
options.register('outputFile', '', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 'if not empty, dump the payload content to this text file')
options.parseArguments()

process = cms.Process("SiPixelClusterShapeLimitsReader")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1), reportEvery = cms.untracked.int32(1)),
    SiPixelClusterShapeLimitsReader = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.run),
    lastValue = cms.uint64(options.run),
    interval = cms.uint64(1)
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:' + options.inputDB

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelClusterShapeLimitsRcd'),
        tag = cms.string(options.inputTag)
    ))
)

from CondTools.SiPixel.siPixelClusterShapeLimitsReader_cfi import siPixelClusterShapeLimitsReader
process.reader = siPixelClusterShapeLimitsReader.clone(
    outputFile = options.outputFile,
)

process.p = cms.Path(process.reader)
