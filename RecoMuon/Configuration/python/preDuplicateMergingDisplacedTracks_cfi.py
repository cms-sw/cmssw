import FWCore.ParameterSet.Config as cms
#only merge muonSeededTracksInOut and muonSeededTracksOutInDisplaced.
import RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi 
preDuplicateMergingDisplacedTracks =  RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi.earlyGeneralTracks.clone(
    TrackProducers = cms.VInputTag(
        #cms.InputTag("earlyGeneralTracks"), 
        cms.InputTag("muonSeededTracksInOut"),
        cms.InputTag("muonSeededTracksOutInDisplaced"),
    ),
    #hasSelector = cms.vint32(0,1,1),
    hasSelector = cms.vint32(1,1),
    selectedTrackQuals = cms.VInputTag(
       # cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"), # not used but needed
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"), 
        cms.InputTag("muonSeededTracksOutInDisplacedSelector","muonSeededTracksOutInDisplacedHighPurity"), 
    ),
    mvaValueTags = cms.VInputTag(
        #cms.InputTag("earlyGeneralTracks","MVAVals"),
        cms.InputTag("muonSeededTracksInOutSelector","MVAVals"), 
        cms.InputTag("muonSeededTracksOutInDisplacedSelector","MVAVals"), 
    ),
    #setsToMerge = cms.VPSet(cms.PSet(pQual = cms.bool(False), tLists = cms.vint32(0, 1,2))),
    setsToMerge = cms.VPSet(cms.PSet(pQual = cms.bool(False), tLists = cms.vint32(1,2))),
    FoundHitBonus  = 100.0,
    LostHitPenalty =   1.0,
)


