import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *

preDuplicateMergingGeneralTracks = TrackCollectionMerger.clone()
preDuplicateMergingGeneralTracks.trackProducers = [
    "earlyGeneralTracks", 
    "muonSeededTracksInOut",
    "muonSeededTracksOutIn",
    ]
preDuplicateMergingGeneralTracks.inputClassifiers =[
   "earlyGeneralTracks", 
   "muonSeededTracksInOutClassifier",
   "muonSeededTracksOutInClassifier"
   ]

preDuplicateMergingGeneralTracks.foundHitBonus  = 100.0
preDuplicateMergingGeneralTracks.lostHitPenalty =   1.0


