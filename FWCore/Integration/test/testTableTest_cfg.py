import FWCore.ParameterSet.Config as cms

process = cms.Process("TABLETEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2))

anInts = [1,2,3]
aFloats = [4.,5., 6.]
aStrings =["einie", "meanie", "meinie"]

process.tableTest = cms.EDProducer("edmtest::TableTestProducer",
                                   anInts = cms.vint32(*anInts),
                                   aFloats = cms.vdouble(*aFloats),
                                   aStrings = cms.vstring(*aStrings) )

process.checkTable = cms.EDAnalyzer("edmtest::TableTestAnalyzer",
                                    table = cms.untracked.InputTag("tableTest"),
                                    anInts = cms.untracked.vint32(*anInts),
                                    aFloats = cms.untracked.vdouble(*aFloats),
                                    aStrings = cms.untracked.vstring(*aStrings) )

process.eventContent = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(process.checkTable, cms.Task(process.tableTest) )
#process.p = cms.Path(process.tableTest+process.eventContent+process.checkTable)

#process.add_(cms.Service("Tracer", dumpPathsAndConsumes= cms.untracked.bool(True) ) )

#process.add_(cms.Service("InitRootHandlers", DebugLevel = cms.untracked.int32(10)))
