import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the GEM geometry model.
#
GEMGeometryESModule = cms.ESProducer("GEMGeometryESModule",
    fromDDD = cms.bool(False),
    fromDD4hep = cms.bool(False),
    appendToDataLabel = cms.string(''),
    alignmentsLabel = cms.string(''),
    applyAlignment = cms.bool(False)
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM

run3_GEM.toModify(GEMGeometryESModule, applyAlignment = True)
phase2_GEM.toModify(GEMGeometryESModule, applyAlignment = False)
