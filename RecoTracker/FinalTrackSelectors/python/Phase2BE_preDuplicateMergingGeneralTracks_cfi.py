import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi 
preDuplicateMergingGeneralTracks =  RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi.earlyGeneralTracks.clone(
    TrackProducers = cms.VInputTag(
        cms.InputTag("earlyGeneralTracks"), 
        cms.InputTag("muonSeededTracksInOut")
    ),
    hasSelector = cms.vint32(0,1),
    selectedTrackQuals = cms.VInputTag(
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"), # not used but needed
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity")
    ),
    setsToMerge = cms.VPSet(cms.PSet(pQual = cms.bool(False), tLists = cms.vint32(0, 1))),
    FoundHitBonus  = 100.0,
    LostHitPenalty =   1.0
)


