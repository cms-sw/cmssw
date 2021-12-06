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
    recordName = cms.string('GadgetRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
process.esprod = cms.ESProducer("DoodadESProducer")

process.looper = cms.Looper("DoodadEDLooper")

process.p1 = cms.Path(process.intEventProducer)
