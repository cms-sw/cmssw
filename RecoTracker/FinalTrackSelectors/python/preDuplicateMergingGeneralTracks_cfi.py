import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
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


# For Phase1PU70
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import trackListMerger as _trackListMerger
eras.trackingPhase1PU70.toReplaceWith(preDuplicateMergingGeneralTracks, _trackListMerger.clone(
    TrackProducers = [
        "earlyGeneralTracks",
        "muonSeededTracksInOut",
        "muonSeededTracksOutIn",
    ],
    hasSelector = [0,1,1],
    selectedTrackQuals = [
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"), # not used but needed
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"),
        cms.InputTag("muonSeededTracksOutInSelector","muonSeededTracksOutInHighPurity"),
    ],
    mvaValueTags = cms.VInputTag(
        cms.InputTag("earlyGeneralTracks","MVAVals"),
        cms.InputTag("muonSeededTracksInOutSelector","MVAVals"),
        cms.InputTag("muonSeededTracksOutInSelector","MVAVals"),
    ),
    setsToMerge = [cms.PSet(pQual = cms.bool(True), tLists = cms.vint32(0, 1,2))],
    FoundHitBonus  = 100.0,
    LostHitPenalty =   1.0,
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
))
