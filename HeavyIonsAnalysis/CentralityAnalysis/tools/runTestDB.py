import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')

ivars.register ('outputTag',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")
ivars.outputTag="HFhits40_MC_Hydjet2760GeV_MC_3XY_V24_v0"

ivars.register ('inputFile',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")

ivars.register ('outputFile',
                mult=ivars.multiplicity.singleton,
                mytype=ivars.varType.string,
                info="for testing")

ivars.inputFile="../data/CentralityTables.root"
ivars.outputFile="Test.db"

ivars.parseArguments()

hiRecord = 'HeavyIonRcd'

import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMMY')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO')
    ),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string("runnumber"),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )


process.makeCentralityTableDB = cms.EDAnalyzer('CentralityPopConProducer',
                                               Source = cms.PSet(makeDBFromTFile = cms.untracked.bool(True),
                                                                 inputFile = cms.string(ivars.inputFile),
                                                                 rootTag = cms.string(ivars.outputTag)
                                                                 ),
                                               record = cms.string(hiRecord),
                                               name= cms.untracked.string(ivars.outputTag),
                                               loggingOn = cms.untracked.bool(True)
                                               )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = "oracle://cms_orcoff_prep/CMS_COND_PHYSICSTOOLS"
process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(3)
process.CondDBCommon.DBParameters.authenticationPath = "authPath"

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          logconnect = cms.untracked.string("sqlite_file:" + "LogsTest.db"),
                                          timetype = cms.untracked.string("runnumber"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string(hiRecord),
                                                                     tag = cms.string(ivars.outputTag)
                                                                     )
                                                            )
                                          )




process.step  = cms.Path(process.makeCentralityTableDB)






