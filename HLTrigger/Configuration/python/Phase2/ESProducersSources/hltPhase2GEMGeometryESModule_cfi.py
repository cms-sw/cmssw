import FWCore.ParameterSet.Config as cms

from Geometry.GEMGeometryBuilder.gemGeometryDB_cfi import (
    GEMGeometryESModule as _GEMGeometryESModule,
)

hltPhase2GEMGeometryESModule = _GEMGeometryESModule.clone(
    fromDD4Hep = False,
)
