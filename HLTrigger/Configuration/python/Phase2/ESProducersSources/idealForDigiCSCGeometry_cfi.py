import FWCore.ParameterSet.Config as cms

from Geometry.CSCGeometryBuilder.cscGeometry_cfi import (
    CSCGeometryESModule as _CSCGeometryESModule,
)

hltPhase2idealForDigiCSCGeometry = _CSCGeometryESModule.clone(
    alignmentsLabel="fakeForIdeal",
    appendToDataLabel="idealForDigi",
    applyAlignment=False,
    useGangedStripsInME1a=False,
)
