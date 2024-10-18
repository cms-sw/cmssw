import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
process = cms.Process('test')

options = VarParsing.VarParsing()
supported_timetypes = {"timestamp", "lumiid"}
options.register( 'db'
                , 'sqlite_file:lhcinfoperls_pop_test.db' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads are going to be read from"
                  )
options.register( 'tag'
                , 'LHCInfoPerLS_PopCon_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag to read from in source db"
                  )
options.register( 'iov'
                , 1
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "IOV: specifies the point in time to read the payload at"
                  )
options.register( 'timetype'
                , 'timestamp' #default
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , f"timetype of the provided IOV, accepted values: {supported_timetypes}"
                  )
options.register( 'csv'
                , False
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.bool
                , "Weather or not to print the values in csv format"
                  )
options.register( 'header'
                , False
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.bool
                , "Weather or not to print header for the csv, works only in when csv=True"
                  )
options.register( 'separator'
                , ','
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "separator for the csv format, works only in when csv=True"
                  )
options.parseArguments()

# TODO check validity of the rest of the arguments
if options.timetype not in supported_timetypes:
  print(f"Provided timetype '{options.timetype}' is not supported (accepted values: {supported_timetypes})", file=sys.stderr)
  exit(1)


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
    timetype = cms.string(options.timetype),
    firstValue = cms.uint64(options.iov),
    lastValue = cms.uint64(options.iov),
    interval = cms.uint64(1)
)

# load info from database
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = options.db # SQLite input

process.PoolDBESSource = cms.ESSource('PoolDBESSource',
    process.CondDB,
    DumpStats = cms.untracked.bool(True),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('LHCInfoPerLSRcd'),
            tag = cms.string(options.tag)
        )
    )
)

process.LHCInfoPerLSAnalyzer = cms.EDAnalyzer('LHCInfoPerLSAnalyzer',
                                              csvFormat = cms.untracked.bool(options.csv),
                                              header = cms.untracked.bool(options.header),
                                              iov = cms.untracked.uint64(options.iov),
                                              separator = cms.untracked.string(options.separator),
                                              
)

process.path = cms.Path(
    process.LHCInfoPerLSAnalyzer
)
