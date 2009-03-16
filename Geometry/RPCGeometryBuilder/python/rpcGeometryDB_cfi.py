import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the RPC geometry model.
#
RPCGeometryESModule = cms.ESProducer("RPCGeometryESModule",
    compatibiltyWith11 = cms.untracked.bool(True),
    useDDD = cms.untracked.bool(False)
)


