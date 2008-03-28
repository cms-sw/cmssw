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
l1NonIsoEgammaRegionalCkfTrackCandidates = copy.deepcopy(ckfTrackCandidates)
#bool   seedCleaning         = false
l1NonIsoEgammaRegionalCkfTrackCandidates.SeedProducer = 'l1NonIsoEgammaRegionalPixelSeedGenerator'
l1NonIsoEgammaRegionalCkfTrackCandidates.SeedLabel = ''
l1NonIsoEgammaRegionalCkfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
l1NonIsoEgammaRegionalCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
# set it as "none" to avoid redundant seed cleaner
l1NonIsoEgammaRegionalCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
#replace l1NonIsoEgammaRegionalCkfTrackCandidates.RedundantSeedCleaner  = "none"
l1NonIsoEgammaRegionalCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'

