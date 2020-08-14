import FWCore.ParameterSet.Config as cms

from Geometry.CSCGeometryBuilder.cscGeometry_cfi import (
    CSCGeometryESModule as _CSCGeometryESModule,
)

### This is the only Geometry EventSetup that changes.
hltPhase2CSCGeometryESModule = _CSCGeometryESModule.clone(useGangedStripsInME1a=False)
