import FWCore.ParameterSet.Config as cms

process = cms.Process("ReadTest")
process.source = cms.Source( "EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))


process.tester = cms.EDAnalyzer("EventSetupIntProductAnalyzer",
                                expectedValues = cms.untracked.vint32(0) 
)

process.dump = cms.EDAnalyzer("PrintEventSetupContent")

process.add_(cms.ESSource("FWLiteESSource", fileName = cms.string("cond_test.root") ))

process.tst = cms.EndPath(process.dump+process.tester)

