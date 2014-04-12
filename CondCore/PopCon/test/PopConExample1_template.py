import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('sinceTime',
                 4294967294, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "SinceTime; default gives latest IOV")
options.register('numberObj',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of objects to write")
options.register('outOfOrder',
                 False, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "append out of order")
options.register('closeIOV',
                 False, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "close the IOV sequence at the last inserted since time")
options.register('tag',
                 'Example_tag', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "output tag")
options.register('connect',
                 'sqlite_file:pop_test.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "connection string")
options.parseArguments()


import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                                  destinations=cms.untracked.vstring("cout"),
                                  cout=cms.untracked.PSet(
                                                          #treshold=cms.untracked.string("INFO")
                                                          )
                                  )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string(options.connect)
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(3)

process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          closeIOV =  cms.untracked.bool(bool(options.closeIOV)),
                                          outOfOrder = cms.untracked.bool(bool(options.outOfOrder)),
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('ThisJob'),
                                                                     tag = cms.string(options.tag)
                                                                     )
                                                            ),
                                          logconnect= cms.untracked.string('sqlite_file:log.db')                                  
                                          )

process.mytest = cms.EDAnalyzer("ExPopConAnalyzer",
                                record = cms.string('ThisJob'),
                                loggingOn= cms.untracked.bool(True),
                                Source=cms.PSet(firstSince=cms.untracked.int64(options.sinceTime),
                                                number=cms.untracked.int64(options.numberObj)
                                                )                            
                                )

process.p = cms.Path(process.mytest)
