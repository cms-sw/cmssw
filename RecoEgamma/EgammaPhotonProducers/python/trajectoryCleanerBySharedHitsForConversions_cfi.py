import FWCore.ParameterSet.Config as cms

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import *
import  TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi
TrajectoryCleanerBySharedHitsForConversions = TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi.trajectoryCleanerBySharedHits.clone(
    ComponentName  = 'TrajectoryCleanerBySharedHitsForConversions',
    fractionShared = 0.5
)
