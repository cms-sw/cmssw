import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')
ivars.files = 'dcache:/pnfs/cmsaf.mit.edu/t2bat/cms/store/user/yetkin/sim/CMSSW_3_3_5/Pythia_MinBias_D6T_900GeV_d20091208/Vertex1207/Pythia_MinBias_D6T_900GeV_d20091208_000005.root'

ivars.output = 'bambu.root'
ivars.maxEvents = -1

ivars.parseArguments()

import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMMY')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(13))
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string("runnumber"),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(3),
                            interval = cms.uint64(3)
                            )

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('centralityfile.root')
                                   )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = "sqlite_file:CentralityTables.db"
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDBCommon,
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('HeavyIonRcd'),
                                                                 tag = cms.string('TestTag_A01')
                                                                 )
                                                        )
                                      )

process.makeCentralityTableTFile = cms.EDAnalyzer('CentralityTableProducer',
                                                  makeDBFromTFile = cms.untracked.bool(False),
                                                  makeTFileFromDB = cms.untracked.bool(True),
                                                  inputTFile = cms.string("/net/hisrv0001/home/yetkin/CMSSW_3_5_2/src/RecoHI/HiCentralityAlgos/macros/test.root"),
                                                  rootTag = cms.string("HFhitBins"),
                                                  nBins = cms.int32(20)
                                                  )

process.step  = cms.Path(process.makeCentralityTableTFile)
    




