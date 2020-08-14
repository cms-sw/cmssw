import FWCore.ParameterSet.Config as cms

from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import (
    trackerGeometry as _trackerGeometry,
)

hltPhase2trackerGeometry = _trackerGeometry.clone(applyAlignment=False)
