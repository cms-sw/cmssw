import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

process.pInt = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
    )

#Dummy looper will loop twice
process.looper = cms.Looper("TestModuleChangeLooper",
    startingValue = cms.untracked.int32(1),
    tag = cms.untracked.InputTag("pInt")
)

process.p1 = cms.Path(process.pInt)
