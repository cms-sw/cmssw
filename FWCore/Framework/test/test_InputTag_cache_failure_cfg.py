import FWCore.ParameterSet.Config as cms
process = cms.Process("Test")

#we want to continue processing after a 'ProductNotFound' exception in order
# to test what happens to the cache held by the InputTag
process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("EmptySource",
    timeBetweenEvents = cms.untracked.uint32(10),
    firstTime = cms.untracked.uint32(1000000)
)

process.double = cms.EDProducer("ToyDoubleProducer", 
                                dvalue = cms.double(1.0)
)

process.doubleTwo = cms.EDProducer("ToyDoubleProducer", 
                                    dvalue = cms.double(1.0)
)

#Need to add an Int producer whose module label is 'beyond' what IntTestAnalyzer is looking for
process.zInt = cms.EDProducer("IntProducer", ivalue=cms.int32(26))

process.getOne = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(1),
    moduleLabel = cms.untracked.string('one')
)

process.p = cms.Path(process.double+process.doubleTwo+process.zInt+process.getOne)
