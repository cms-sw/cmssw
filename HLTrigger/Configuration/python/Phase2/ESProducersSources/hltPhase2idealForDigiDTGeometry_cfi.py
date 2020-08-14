import FWCore.ParameterSet.Config as cms

from Geometry.DTGeometryBuilder.dtGeometry_cfi import (
    DTGeometryESModule as _DTGeometryESModule,
)

hltPhase2idealForDigiDTGeometry = _DTGeometryESModule.clone(
    alignmentsLabel="fakeForIdeal",
    appendToDataLabel="idealForDigi",
    applyAlignment=False,
)
