import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the ME0 geometry model.
#
ME0GeometryESModule = cms.ESProducer("ME0GeometryESModule",
    useDDD = cms.bool(True),
    use10EtaPart = cms.bool(True),
)
