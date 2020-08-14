import FWCore.ParameterSet.Config as cms

from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import (
    trackerNumberingGeometry as _trackerNumberingGeometry,
)

hltPhase2trackerNumberingGeometry = _trackerNumberingGeometry.clone()
