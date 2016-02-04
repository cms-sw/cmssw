import FWCore.ParameterSet.Config as cms

RPCConeBuilder = cms.ESProducer("RPCConeBuilder",
    towerBeg = cms.int32(0),
    towerEnd = cms.int32(16)
)

rpcconesrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCConeBuilderRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


