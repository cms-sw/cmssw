from FWCore.PythonFramework.CmsRun import CmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

nEvents = 10
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(nEvents))

var = 5
outList = []

process.m = cms.EDProducer("edmtest::PythonTestProducer", inputVariable = cms.string("var"),
                            outputListVariable = cms.string("outList"),
                            source = cms.InputTag("ints"))

process.ints = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.p = cms.Path(process.m, cms.Task(process.ints))

cmsRun = CmsRun(process)

cmsRun.run()

assert (outList == [1]*nEvents)

