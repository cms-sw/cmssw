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
l1IsoElectronsRegionalCkfTrackCandidates = copy.deepcopy(ckfTrackCandidates)
#bool   seedCleaning         = false
l1IsoElectronsRegionalCkfTrackCandidates.SeedProducer = 'l1IsoElectronsRegionalPixelSeedGenerator'
l1IsoElectronsRegionalCkfTrackCandidates.SeedLabel = ''
l1IsoElectronsRegionalCkfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
l1IsoElectronsRegionalCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
# set it as "none" to avoid redundant seed cleaner
l1IsoElectronsRegionalCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
#replace l1IsoElectronsRegionalCkfTrackCandidates.RedundantSeedCleaner  = "none"
l1IsoElectronsRegionalCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'

