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


# For Phase1PU70
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import trackListMerger as _trackListMerger
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toReplaceWith(preDuplicateMergingGeneralTracks, _trackListMerger.clone(
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

# For Phase2PU140
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toReplaceWith(preDuplicateMergingGeneralTracks, _trackListMerger.clone(
    TrackProducers = cms.VInputTag(
        cms.InputTag("earlyGeneralTracks"),
        cms.InputTag("muonSeededTracksInOut"),
        #cms.InputTag("muonSeededTracksOutIn"),
    ),
    hasSelector = cms.vint32(0,1),
    selectedTrackQuals = cms.VInputTag(
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"), # not used but needed
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity")
        #cms.InputTag("muonSeededTracksOutInSelector","muonSeededTracksOutInHighPurity"),
    ),
    mvaValueTags = cms.VInputTag(
        cms.InputTag("earlyGeneralTracks","MVAVals"),
        cms.InputTag("muonSeededTracksInOutSelector","MVAVals"),
    #    cms.InputTag("muonSeededTracksOutInSelector","MVAVals"),
    ),
    setsToMerge = cms.VPSet(cms.PSet(pQual = cms.bool(False), tLists = cms.vint32(0, 1))),
    FoundHitBonus  = 100.0,
    LostHitPenalty =   1.0,
    indivShareFrac = cms.vdouble(1.0, 0.16, 0.095, 0.09, 0.095,0.095, 0.095, 0.08),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
)

