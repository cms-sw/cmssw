import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')

ivars.register ('outputTag',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")
ivars.outputTag="demo"

ivars.register ('inputFile',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")

ivars.register ('outputFile',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")

ivars.inputFile="./input.root"
ivars.outputFile="./output.db"

ivars.parseArguments()

import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMMY')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string("runnumber"),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(200),
                            interval = cms.uint64(1)
                            )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = "sqlite_file:" + ivars.outputFile

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          timetype = cms.untracked.string("runnumber"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('HeavyIonRcd'),
                                                                     tag = cms.string(ivars.outputTag)
                                                                     )
                                                            )
                                          )

process.makeCentralityTableDB = cms.EDAnalyzer('CentralityTableProducer',
                                               makeDBFromTFile = cms.untracked.bool(True),
                                               inputTFile = cms.string(ivars.inputFile),
                                               rootTag = cms.string(ivars.outputTag)
                                               )

process.step  = cms.Path(process.makeCentralityTableDB)
    




