import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

preDuplicateMergingGeneralTracks = TrackCollectionMerger.clone(
    trackProducers   = ["earlyGeneralTracks", 
                        "muonSeededTracksInOut",
                        "muonSeededTracksOutIn"],
    inputClassifiers = ["earlyGeneralTracks", 
                       "muonSeededTracksInOutClassifier",
                       "muonSeededTracksOutInClassifier"],
    foundHitBonus    = 100.0,
    lostHitPenalty   = 1.0
)

from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.ProcessModifiers.displacedRegionalTracking_cff import displacedRegionalTracking
def _extend_displacedRegional(x):
     x.trackProducers += ['displacedRegionalStepTracks']
     x.inputClassifiers += ['displacedRegionalStep']
(trackingPhase1 & displacedRegionalTracking).toModify(preDuplicateMergingGeneralTracks, _extend_displacedRegional)

# For Phase2PU140
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import trackListMerger as _trackListMerger
trackingPhase2PU140.toReplaceWith(preDuplicateMergingGeneralTracks, _trackListMerger.clone(
    TrackProducers     = ["earlyGeneralTracks", 
                          "muonSeededTracksInOut", 
                          "muonSeededTracksOutIn"],
    hasSelector        = [0,1,1],
    selectedTrackQuals = ["muonSeededTracksInOutSelector:muonSeededTracksInOutHighPurity", # not used but needed
                          "muonSeededTracksInOutSelector:muonSeededTracksInOutHighPurity",
                          "muonSeededTracksOutInSelector:muonSeededTracksOutInHighPurity"],
    mvaValueTags = cms.VInputTag(
        cms.InputTag("earlyGeneralTracks","MVAVals"),
        cms.InputTag("muonSeededTracksInOutSelector","MVAVals"),
        cms.InputTag("muonSeededTracksOutInSelector","MVAVals"),
    ),
    setsToMerge = cms.VPSet(cms.PSet(pQual = cms.bool(False), tLists = cms.vint32(0, 1, 2))),
    FoundHitBonus    = 100.0,
    LostHitPenalty   = 1.0,
    indivShareFrac   = [1.0, 0.16, 0.095, 0.09, 0.095,0.095, 0.095, 0.08],
    copyExtras       = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
)
