import FWCore.ParameterSet.Config as cms

hltGeneralTracks = cms.EDProducer("TrackListMerger",
    Epsilon = cms.double(-0.001),
    FoundHitBonus = cms.double(5.0),
    LostHitPenalty = cms.double(5.0),
    MaxNormalizedChisq = cms.double(1000.0),
    MinFound = cms.int32(3),
    MinPT = cms.double(0.9),
    ShareFrac = cms.double(0.19),
    TrackProducers = cms.VInputTag("hltInitialStepTrackSelectionHighPurity"),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(True),
    copyMVA = cms.bool(False),
    hasSelector = cms.vint32(0),
    indivShareFrac = cms.vdouble(0.1),
    makeReKeyedSeeds = cms.untracked.bool(False),
    newQuality = cms.string('confirmed'),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hltInitialStepTrackSelectionHighPurity")),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0)
    )),
    trackAlgoPriorityOrder = cms.string('trackAlgoPriorityOrder'),
    writeOnlyTrkQuals = cms.bool(False)
)


from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
_hltGeneralTracksLegacy = hltGeneralTracks.clone(
    TrackProducers = ["hltInitialStepTrackSelectionHighPurity", "hltHighPtTripletStepTrackSelectionHighPurity"],
    hasSelector = [0, 0],
    indivShareFrac = [1.0, 1.0],
    selectedTrackQuals = ["hltInitialStepTrackSelectionHighPurity", "hltHighPtTripletStepTrackSelectionHighPurity"],
    setsToMerge = {0: dict(pQual=True, tLists=[0, 1])}
)
hltPhase2LegacyTracking.toReplaceWith(hltGeneralTracks, _hltGeneralTracksLegacy)


from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from ..modules.hltPhase2PixelTracks_cfi import *
_hltGeneralTracksNGTScoutingLST = hltGeneralTracks.clone(
    TrackProducers = ["hltPhase2PixelTracks", "hltInitialStepTracksT4T5TCLST"],
    hasSelector = [0,0],
    indivShareFrac = [0.1,0.1],
    selectedTrackQuals = ["hltPhase2PixelTracks", "hltInitialStepTracksT4T5TCLST"],
    setsToMerge = {0: dict(pQual=True, tLists=[0,1])}
)

(ngtScouting & ~trackingLST).toReplaceWith(hltGeneralTracks, hltPhase2PixelTracks)

(ngtScouting & trackingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksNGTScoutingLST)
