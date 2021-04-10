import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

process.intEventProducer = cms.EDProducer("IntProducer",
    ivalue = cms.int32(42)
)

process.essource = cms.ESSource("EmptyESSource",
    recordName = cms.string('DummyRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
process.add_(cms.ESProducer("LoadableDummyProvider",
                            value = cms.untracked.int32(5)))

process.looper = cms.Looper("IntTestLooper",
    srcEvent = cms.untracked.VInputTag("intEventProducer"),
    expectEventValues = cms.untracked.vint32(42),
    expectESValue = cms.untracked.int32(5)
)

process.p1 = cms.Path(process.intEventProducer)
