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
l1NonIsoElectronsRegionalCkfTrackCandidates = copy.deepcopy(ckfTrackCandidates)
#bool   seedCleaning         = false
l1NonIsoElectronsRegionalCkfTrackCandidates.SeedProducer = 'l1NonIsoElectronsRegionalPixelSeedGenerator'
l1NonIsoElectronsRegionalCkfTrackCandidates.SeedLabel = ''
l1NonIsoElectronsRegionalCkfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
l1NonIsoElectronsRegionalCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
# set it as "none" to avoid redundant seed cleaner
l1NonIsoElectronsRegionalCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
#replace l1NonIsoElectronsRegionalCkfTrackCandidates.RedundantSeedCleaner  = "none"
l1NonIsoElectronsRegionalCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'

