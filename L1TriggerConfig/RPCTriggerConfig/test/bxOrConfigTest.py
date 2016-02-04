import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.rpcconfsrc1 = cms.ESProducer("RPCTriggerBxOrConfig",
    firstBX = cms.int32(-2),
     lastBX = cms.int32(0)
)

process.rpcconfsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCBxOrConfigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.ms = cms.EDFilter("TestBxOrConfig")

process.p = cms.Path(process.ms)


