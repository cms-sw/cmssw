import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#
# This cfi should be included to run the CkfTrackCandidateMaker 
#
hltEgammaRegionalCkfTrackCandidates = copy.deepcopy(ckfTrackCandidates)
#bool   seedCleaning         = false
hltEgammaRegionalCkfTrackCandidates.src = 'hltEgammaRegionalPixelSeedGenerator'
hltEgammaRegionalCkfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
hltEgammaRegionalCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
# set it as "none" to avoid redundant seed cleaner
hltEgammaRegionalCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
#replace hltEgammaRegionalCkfTrackCandidates.RedundantSeedCleaner  = "none"
hltEgammaRegionalCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'

