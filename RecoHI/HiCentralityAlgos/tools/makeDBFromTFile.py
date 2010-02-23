import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')

ivars.register ('randomNumber',
                mult=ivars.multiplicity.singleton,
                info="for testing")
ivars.randomNumber=1

ivars.parseArguments()

import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMMY')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string("runnumber"),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = "sqlite_file:CentralityTables.db"

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          timetype = cms.untracked.string("runnumber"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('HeavyIonRcd'),
                                                                     tag = cms.string('HFhitSum_Hydjet4TeV_352_17kE_Test01')
                                                                     )
                                                            )
                                          )

process.makeCentralityTableDB = cms.EDAnalyzer('CentralityTableProducer',
                                               makeDBFromTFile = cms.untracked.bool(True),
                                               inputTFile = cms.string("/net/hisrv0001/home/yetkin/CMSSW_3_5_2/src/RecoHI/HiCentralityAlgos/macros/bins20_4TeV_CMSSW_3_5_2.root"),
                                               rootTag = cms.string("HFhitBins"),
                                               nBins = cms.int32(20)
                                               )

process.step  = cms.Path(process.makeCentralityTableDB)
    




