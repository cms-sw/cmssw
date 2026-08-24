import FWCore.ParameterSet.Config as cms

# Expected behavior is that intprod2 is deleted because it is not consumed by any other module, and is not visible in
# the wantSummary report.

process = cms.Process("test")

process.maxEvents.input = 1
process.options.wantSummary = True

process.source = cms.Source("EmptySource")

process.intprod = cms.EDProducer("IntProducer", ivalue=cms.int32(1))
process.intprod2 = cms.EDProducer("IntProducer", ivalue=cms.int32(1))
process.intprod3 = cms.EDProducer("IntProducer", ivalue=cms.int32(1))

process.t = cms.Task(
    process.intprod2
)

process.p = cms.Path(
    process.intprod+
    process.intprod3,
    process.t
)
