import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')

options.register ('input', # input text file with encoded weight groups                             
                'EcalTPGOddWeightGroup.txt', 
                VarParsing.VarParsing.multiplicity.singleton, 
                VarParsing.VarParsing.varType.string,          
                "input")           
options.register ('output', # output file with SQLite format                              
                'EcalTPGOddWeightGroup.db', 
                VarParsing.VarParsing.multiplicity.singleton, 
                VarParsing.VarParsing.varType.string,          
                "output")
options.register ('filetype', # input file format txt/xml                              
                'txt', 
                VarParsing.VarParsing.multiplicity.singleton, 
                VarParsing.VarParsing.varType.string,          
                "filetype")
options.register('outputtag',
                 'EcalTPGOddWeightGroup',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "outputtag")
options.parseArguments()


process = cms.Process("ProcessOne")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100000000000),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(100000000000),
    interval = cms.uint64(1)
)

process.load("CondCore.CondDB.CondDB_cfi")

process.CondDB.connect = 'sqlite_file:%s'%(options.output)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB, 
  logconnect = cms.untracked.string('sqlite_file:log.db'),   
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalTPGOddWeightGroupRcd'),
      tag = cms.string(options.outputtag)
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalTPGOddWeightGroupAnalyzer",
  record = cms.string('EcalTPGOddWeightGroupRcd'),
  loggingOn= cms.untracked.bool(True),
  IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
  SinceAppendMode=cms.bool(True),
  Source=cms.PSet(
    firstRun = cms.string('1'),
    lastRun = cms.string('10'),
    OnlineDBSID = cms.string(''),
    OnlineDBUser = cms.string(''),
    OnlineDBPassword = cms.string(''),
    LocationSource = cms.string(''),
    Location = cms.string(''),
    GenTag = cms.string(''),
    RunType = cms.string(''),
    fileType = cms.string(options.filetype),
    fileName = cms.string(options.input),
  )
)

process.p = cms.Path(process.Test1)
