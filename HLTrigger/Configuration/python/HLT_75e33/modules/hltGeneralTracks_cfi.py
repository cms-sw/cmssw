import FWCore.ParameterSet.Config as cms

hltGeneralTracks = cms.EDProducer("TrackListMerger",
    Epsilon = cms.double(-0.001),
    FoundHitBonus = cms.double(5.0),
    LostHitPenalty = cms.double(5.0),
    MaxNormalizedChisq = cms.double(1000.0),
    MinFound = cms.int32(3),
    MinPT = cms.double(0.8),
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
    MinPT = cms.double(0.9),
    TrackProducers = ["hltInitialStepTrackSelectionHighPurity", "hltHighPtTripletStepTrackSelectionHighPurity"],
    hasSelector = [0, 0],
    indivShareFrac = [1.0, 1.0],
    selectedTrackQuals = ["hltInitialStepTrackSelectionHighPurity", "hltHighPtTripletStepTrackSelectionHighPurity"],
    setsToMerge = {0: dict(pQual=True, tLists=[0, 1])}
)
hltPhase2LegacyTracking.toReplaceWith(hltGeneralTracks, _hltGeneralTracksLegacy)


from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST

_hltGeneralTracksNGTScouting = cms.EDProducer("RecoTrackSelector",
                                              src = cms.InputTag('hltPhase2PixelTracks'),
                                              copyExtras = cms.untracked.bool(True),
                                              passThrough = cms.bool(True),
                                              usePV = cms.bool(False),
                                              minHit = cms.int32(0),
                                              minLayer = cms.int32(0),
                                              min3DLayer = cms.int32(0),
                                              minPixelHit = cms.int32(0),
                                              ptMin = cms.double(0.0),
                                              maxChi2 = cms.double(10000.0),
                                              beamSpot = cms.InputTag('hltOnlineBeamSpot'),
                                              vertexTag = cms.InputTag(''),
                                              tip = cms.double(1e9),
                                              lip = cms.double(1e9),
                                              minRapidity = cms.double(-9.9),
                                              maxRapidity = cms.double(9.9),
                                              minPhi = cms.double(-3.2),
                                              maxPhi = cms.double(3.2),
                                              quality = cms.vstring(),
                                              algorithm = cms.vstring())

_hltGeneralTracksNGTScoutingLST = hltGeneralTracks.clone(
    MinPT = cms.double(0.9),
    TrackProducers = ["hltPhase2PixelTracks", "hltInitialStepTracksT4T5TCLST"],
    hasSelector = [0,0],
    indivShareFrac = [0.1,0.1],
    selectedTrackQuals = ["hltPhase2PixelTracks", "hltInitialStepTracksT4T5TCLST"],
    setsToMerge = {0: dict(pQual=True, tLists=[0,1])}
)

(ngtScouting & ~trackingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksNGTScouting)
(ngtScouting & trackingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksNGTScoutingLST)
