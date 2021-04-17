import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the RPC geometry model.
#
RPCGeometryESModule = cms.ESProducer("RPCGeometryESModule",
    fromDDD = cms.untracked.bool(False),
    fromDD4hep = cms.untracked.bool(False)
)


