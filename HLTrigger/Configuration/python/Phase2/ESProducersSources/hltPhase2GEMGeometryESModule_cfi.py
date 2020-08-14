import FWCore.ParameterSet.Config as cms

from Geometry.GEMGeometryBuilder.gemGeometry_cfi import (
    GEMGeometryESModule as _GEMGeometryESModule,
)

hltPhase2GEMGeometryESModule = _GEMGeometryESModule.clone()
