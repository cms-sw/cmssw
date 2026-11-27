import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonIsoTracks = cms.EDProducer("TrackListMerger",
    Epsilon = cms.double(-0.001),
    FoundHitBonus = cms.double(5.0),
    LostHitPenalty = cms.double(5.0),
    MaxNormalizedChisq = cms.double(1000.0),
    MinFound = cms.int32(3),
    MinPT = cms.double(0.9),
    ShareFrac = cms.double(0.19),
    TrackProducers = cms.VInputTag("hltPhase2L3MuonInitialStepTracksSelectionHighPurity", "hltPhase2L3MuonHighPtTripletStepTracksSelectionHighPurity"),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(True),
    copyMVA = cms.bool(False),
    hasSelector = cms.vint32(0, 0),
    indivShareFrac = cms.vdouble(1.0, 1.0),
    makeReKeyedSeeds = cms.untracked.bool(False),
    newQuality = cms.string('confirmed'),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hltPhase2L3MuonInitialStepTracksSelectionHighPurity"), cms.InputTag("hltPhase2L3MuonHighPtTripletStepTracksSelectionHighPurity")),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0, 1)
    )),
    trackAlgoPriorityOrder = cms.string('hltPhase2L3MuonTrackAlgoPriorityOrder'),
    writeOnlyTrkQuals = cms.bool(False)
)

_hltPhase2L3MuonIsoTracksPixelSelector = cms.EDProducer("TrackSelectorByRegion",
    produceMask = cms.bool(False),
    produceTrackCollection = cms.bool(True),
    regions = cms.InputTag("hltPhase2L3MuonPixelTracksAndHighPtTripletTrackingRegions"),
    tracks = cms.InputTag("hltPhase2PixelTracks")
)

from Configuration.ProcessModifiers.phase2MuonPixelTracksSelector_cff import phase2MuonPixelTracksSelector
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from Configuration.ProcessModifiers.phase2MuonSeedsSelector_cff import phase2MuonSeedsSelector
(phase2MuonPixelTracksSelector | ngtScouting | phase2MuonSeedsSelector).toReplaceWith(
    hltPhase2L3MuonIsoTracks,
    _hltPhase2L3MuonIsoTracksPixelSelector
)
