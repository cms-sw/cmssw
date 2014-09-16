import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the GEM geometry model.
#
GEMGeometryESModule = cms.ESProducer("GEMGeometryESModule",
    compatibiltyWith11 = cms.untracked.bool(True),
    useDDD = cms.untracked.bool(True)
)


