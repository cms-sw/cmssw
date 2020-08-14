import FWCore.ParameterSet.Config as cms

from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import (
    trackerGeometry as _trackerGeometry,
)

hltPhase2idealForDigiTrackerGeometry = _trackerGeometry.clone(
    alignmentsLabel="fakeForIdeal",
    appendToDataLabel="idealForDigi",
    applyAlignment=False,
)
