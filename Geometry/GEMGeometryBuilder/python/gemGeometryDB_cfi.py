import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the GEM geometry model.
#
GEMGeometryESModule = cms.ESProducer("GEMGeometryESModule",
    fromDDD = cms.bool(False),
    fromDD4Hep = cms.bool(False),
    alignmentsLabel = cms.string(''),
    applyAlignment = cms.bool(False)
)
