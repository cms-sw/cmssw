import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *

preDuplicateMergingDisplacedTracks = TrackCollectionMerger.clone()
preDuplicateMergingDisplacedTracks.trackProducers = [
    "muonSeededTracksInOut",
    "muonSeededTracksOutInDisplaced",
    ]
preDuplicateMergingDisplacedTracks.inputClassifiers =[
   "muonSeededTracksInOutClassifier",
   "muonSeededTracksOutInDispacedClassifier"
   ]

preDuplicateMergingDisplacedTracks.foundHitBonus  = 100.0
preDuplicateMergingDisplacedTracks.lostHitPenalty =   1.0

