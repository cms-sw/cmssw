import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# ----- Options -----
options = VarParsing('analysis')
options.register(
    'size',
    0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "Size parameter for LHCInfoPerFillWriter"
)
options.register(
    'db',
    'sqlite_file:LHCInfoPerFill.sqlite',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Database connection string"
)
options.parseArguments()

# ----- Process -----
process = cms.Process('test')

process.source = cms.Source('EmptyIOVSource',
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# DB service
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = options.db

process.PoolDBOutputService = cms.Service('PoolDBOutputService',
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('LHCInfoPerFillRcd'),
            tag = cms.string('LHCInfoPerFillFake'),
        )
    )
)

# Analyzer with option passed in
process.LHCInfoPerFillWriter = cms.EDAnalyzer(
    'LHCInfoPerFillWriter',
    size = cms.untracked.int32(options.size)
)

process.path = cms.Path(process.LHCInfoPerFillWriter)
