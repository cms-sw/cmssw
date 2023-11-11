import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
process = cms.Process('test')

options = VarParsing.VarParsing()
supported_timetypes = {"timestamp", "lumiid"}
options.register( 'db'
                , 'frontier://FrontierProd/CMS_CONDITIONS' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads are going to be read from"
                  )
options.register( 'tag'
                , None #default value is None because this argument is required
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
options.parseArguments()

if options.tag == None:
  print(f"Please specify the tag by adding to the command: tag=<tag to be printed>", file=sys.stderr)
  exit(1)

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
process.CondDB.connect= options.db # SQLite input

process.PoolDBESSource = cms.ESSource('PoolDBESSource',
    process.CondDB,
    DumpStat = cms.untracked.bool(False),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('LHCInfoPerFillRcd'),
            tag = cms.string(options.tag)
        )
    )
)

process.LHCInfoPerFillAnalyzer = cms.EDAnalyzer('LHCInfoPerFillAnalyzer',
                                                iov = cms.untracked.uint64(options.iov)
)

process.path = cms.Path(
    process.LHCInfoPerFillAnalyzer
)
