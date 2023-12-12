import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
process = cms.Process('test')

options = VarParsing.VarParsing()
options.register( 'db'
                , 'sqlite_file:lhcinfoperfill_pop_test.db' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads are going to be read from"
                  )
options.register( 'tag'
                , 'LHCInfoPerFill_PopCon_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag to read from in source db"
                  )
options.register( 'timestamp'
                , 1
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Timestamp to which payload with relavant IOV will be read"
                  )
options.parseArguments()

# minimum logging
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.source = cms.Source('EmptyIOVSource',
    timetype = cms.string('timestamp'),
    firstValue = cms.uint64(options.timestamp),
    lastValue = cms.uint64(options.timestamp),
    interval = cms.uint64(1)
)
# load info from database
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect= options.db # SQLite input

process.PoolDBESSource = cms.ESSource('PoolDBESSource',
    process.CondDB,
    DumpStats = cms.untracked.bool(True),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('LHCInfoPerFillRcd'),
            tag = cms.string(options.tag)
        )
    )
)

process.LHCInfoPerFillAnalyzer = cms.EDAnalyzer('LHCInfoPerFillAnalyzer')

process.path = cms.Path(
    process.LHCInfoPerFillAnalyzer
)
