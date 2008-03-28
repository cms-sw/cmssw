import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the DT geometry model.
#
DTGeometryESModule = cms.ESProducer("DTGeometryESModule",
    applyAlignment = cms.untracked.bool(False)
)


