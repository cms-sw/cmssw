import FWCore.ParameterSet.Config as cms

from Geometry.GEMGeometryBuilder.me0Geometry_cfi import (
    ME0GeometryESModule as _ME0GeometryESModule,
)

hltPhase2ME0GeometryESModule = _ME0GeometryESModule.clone()
