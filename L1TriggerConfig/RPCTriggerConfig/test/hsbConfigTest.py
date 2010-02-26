import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.rpcconfsrc1 = cms.ESProducer("RPCTriggerHsbConfig",
    hsb0Mask = cms.vint32(1,2,3,0,1,2,3,0),
    hsb1Mask = cms.vint32(0,1,2,3,0,1,2,3)
)

process.rpcconfsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCHsbConfigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.ms = cms.EDFilter("TestHsbConfig")

process.p = cms.Path(process.ms)


