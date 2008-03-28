import FWCore.ParameterSet.Config as cms

# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# TrackerTrajectoryBuilders
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cff import *
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
l1IsoEgammaRegionalCkfTrackCandidates = copy.deepcopy(ckfTrackCandidates)
#bool   seedCleaning         = false
l1IsoEgammaRegionalCkfTrackCandidates.SeedProducer = 'l1IsoEgammaRegionalPixelSeedGenerator'
l1IsoEgammaRegionalCkfTrackCandidates.SeedLabel = ''
l1IsoEgammaRegionalCkfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
l1IsoEgammaRegionalCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
# set it as "none" to avoid redundant seed cleaner
l1IsoEgammaRegionalCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
#replace l1IsoEgammaRegionalCkfTrackCandidates.RedundantSeedCleaner  = "none"
l1IsoEgammaRegionalCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'

