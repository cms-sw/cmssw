import FWCore.ParameterSet.Config as cms

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import (
    trajectoryCleanerBySharedHits as _trajectoryCleanerBySharedHits,
)

hltPhase2trajectoryCleanerBySharedHits = _trajectoryCleanerBySharedHits.clone()
