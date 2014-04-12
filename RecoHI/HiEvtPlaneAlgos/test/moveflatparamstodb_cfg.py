import FWCore.ParameterSet.VarParsing as VarParsing
import string
import subprocess
import os

if os.access("flatparms_new.db",os.F_OK)==True:
    ret = subprocess.Popen(['rm','flatparms_new.db'])
    ret.wait()
    
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
ivars.outputFile="./flatparms_new.db"

ivars.parseArguments()
import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.CondDBCommon.connect = "sqlite_file:" + ivars.outputFile

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          timetype = cms.untracked.string("runnumber"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('HeavyIonRPRcd'),
                                                                     tag = cms.string('flatParamtest')
                                                                     )
                                                            )
                                          )


process.MoveFlatParamsToDB = cms.EDAnalyzer('MoveFlatParamsToDB',
                                           inputTFile = cms.string(ivars.inputFile),
                                           rootTag = cms.string(ivars.outputTag)
)


process.p = cms.Path(process.MoveFlatParamsToDB)
