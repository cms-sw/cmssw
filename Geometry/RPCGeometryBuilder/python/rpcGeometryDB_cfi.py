import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the RPC geometry model.
#
RPCGeometryESModule = cms.ESProducer("RPCGeometryESModule",
    fromDDD = cms.untracked.bool(False),
    fromDD4hep = cms.untracked.bool(False)
)


# foo bar baz
# xAaktZ3umVEoE
# uliiQ2ly2L0zq
