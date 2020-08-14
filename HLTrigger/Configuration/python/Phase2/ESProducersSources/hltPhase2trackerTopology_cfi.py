import FWCore.ParameterSet.Config as cms

from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import (
    trackerTopology as _trackerTopology,
)

hltPhase2trackerTopology = _trackerTopology.clone()
