import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

preDuplicateMergingDisplacedTracks = TrackCollectionMerger.clone()
preDuplicateMergingDisplacedTracks.trackProducers = [
    "muonSeededTracksInOut",
    "muonSeededTracksOutInDisplaced",
    ]
preDuplicateMergingDisplacedTracks.inputClassifiers =[
   "muonSeededTracksInOutClassifier",
   "muonSeededTracksOutInDisplacedClassifier"
   ]

preDuplicateMergingDisplacedTracks.foundHitBonus  = 100.0
preDuplicateMergingDisplacedTracks.lostHitPenalty =   1.0

# For Phase2PU140 tracking, take out muonSeededTracksInOut because the
# cut-selector module is technically incompatible with this one. Since
# that configuration is indended only for tracking comparisons (not
# for production), it is not worth of the effort to try to fix the
# situation.
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(preDuplicateMergingDisplacedTracks,
    trackProducers = [x for x in preDuplicateMergingDisplacedTracks.trackProducers if x != "muonSeededTracksInOut"],
    inputClassifiers = [x for x in preDuplicateMergingDisplacedTracks.inputClassifiers if x != "muonSeededTracksInOutClassifier"],
)

