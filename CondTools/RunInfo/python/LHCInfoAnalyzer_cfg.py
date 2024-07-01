import sys

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
process = cms.Process('test')

options = VarParsing.VarParsing()
options.register( 'source'
                , 'sqlite_file:lhcinfo_pop_test.db' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads are going to be read from"
                  )
options.register( 'tag'
                , 'LHCInfo_PopCon_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag to read from in source"
                  )
options.register( 'timestamp'
                , 7133428598295232512
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
        threshold = cms.untracked.string('ERROR')
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
process.CondDB.connect= options.source # SQLite input

process.PoolDBESSource = cms.ESSource('PoolDBESSource',
    process.CondDB,
    DumpStats = cms.untracked.bool(True),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('LHCInfoRcd'),
            tag = cms.string(options.tag)
        )
    )
)

process.LHCInfoAnalyzer = cms.EDAnalyzer('LHCInfoAnalyzer')

process.path = cms.Path(
    process.LHCInfoAnalyzer
)