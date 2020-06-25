import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the RPC geometry model.
#
RPCGeometryESModule = cms.ESProducer("RPCGeometryESModule",
    useDDD = cms.untracked.bool(False),
    useDD4hep = cms.untracked.bool(False)
)


