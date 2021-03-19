# Test the skipBadFiles option of PoolSource
import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTRECO")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(3))

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducer") ),
  expectedSum = cms.untracked.int32(54)
)

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            skipBadFiles = cms.untracked.bool(True),
                            skipEvents = cms.untracked.uint32(15), #skips all events in first file
                            fileNames = cms.untracked.vstring('file:PoolInputTest.root',
                                                              'file:this_file_doesnt_exist.root',
                                                              'file:this_file_doesnt_exist.root',
                                                              'file:PoolInputTest.root',
                                                              'file:this_file_doesnt_exist.root',
                                                              'file:this_file_doesnt_exist.root',
                                                              'file:PoolInputTest.root',
                                                              'file:this_file_doesnt_exist.root',
                                                              'file:this_file_doesnt_exist.root')
)

process.p = cms.Path(process.intProducer * process.a1)


