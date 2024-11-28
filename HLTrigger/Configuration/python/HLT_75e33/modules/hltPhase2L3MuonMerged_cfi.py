import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonMerged = cms.EDProducer("TrackListMerger",
    Epsilon = cms.double(-0.001),
    FoundHitBonus = cms.double(5.0),
    LostHitPenalty = cms.double(20.0),
    MaxNormalizedChisq = cms.double(1000.0),
    MinFound = cms.int32(3),
    MinPT = cms.double(0.05),
    ShareFrac = cms.double(0.19),
    TrackProducers = cms.VInputTag(
        "hltPhase2L3OIMuonTrackSelectionHighPurity",
        "hltIter2Phase2L3FromL1TkMuonMerged",
    ),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(True),
    copyMVA = cms.bool(False),
    hasSelector = cms.vint32(0, 0),
    indivShareFrac = cms.vdouble(1.0, 1.0),
    newQuality = cms.string("confirmed"),
    selectedTrackQuals = cms.VInputTag(
        "hltPhase2L3OIMuonTrackSelectionHighPurity",
        "hltIter2Phase2L3FromL1TkMuonMerged",
    ),
    setsToMerge = cms.VPSet(cms.PSet(pQual = cms.bool(False), tLists = cms.vint32(0, 1))),
    trackAlgoPriorityOrder = cms.string("hltESPTrackAlgoPriorityOrder"),
    writeOnlyTrkQuals = cms.bool(False),
)

from Configuration.ProcessModifiers.phase2L2AndL3Muons_cff import phase2L2AndL3Muons
phase2L2AndL3Muons.toModify(
    hltPhase2L3MuonMerged,
    TrackProducers = cms.VInputTag(
        "hltPhase2L3OIMuonTrackSelectionHighPurity",
        "hltPhase2L3MuonFilter:L3IOTracksFiltered",
    ),
    selectedTrackQuals = cms.VInputTag(
        "hltPhase2L3OIMuonTrackSelectionHighPurity",
        "hltPhase2L3MuonFilter:L3IOTracksFiltered",
    ),
)

from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst
(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toModify(
    hltPhase2L3MuonMerged,
    TrackProducers = cms.VInputTag(
        "hltPhase2L3MuonFilter:L3OITracksFiltered",
        "hltIter2Phase2L3FromL1TkMuonMerged",
    ),
    selectedTrackQuals = cms.VInputTag(
        "hltPhase2L3MuonFilter:L3OITracksFiltered",
        "hltIter2Phase2L3FromL1TkMuonMerged",
    ),
)
