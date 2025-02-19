import FWCore.ParameterSet.Config as cms

from RecoTracker.CkfPattern.CkfTrackCandidates_cff import * 
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedSeeds_cfi import *
MaterialPropagator.Mass = 0.139 #pion (default is muon)
OppositeMaterialPropagator.Mass = 0.139

#trajectory filter settings
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
ckfBaseTrajectoryFilter.filterPset.minimumNumberOfHits = 6 #default is 5
ckfBaseTrajectoryFilter.filterPset.minPt = 2.0 #default is 0.9

# trajectory builder settings
CkfTrajectoryBuilder.maxCand = 5 #default is 5
CkfTrajectoryBuilder.intermediateCleaning = False #default is true
CkfTrajectoryBuilder.alwaysUseInvalidHits = False #default is true

### primary track candidates
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiPrimTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
	TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds',
	TrajectoryBuilder = 'CkfTrajectoryBuilder', #instead of GroupedCkfTrajectoryBuilder
	src = 'hiPixelTrackSeeds', 
	RedundantSeedCleaner = 'none',
	doSeedingRegionRebuilding = False 
)


