import FWCore.ParameterSet.Config as cms

hltGeneralTracks = cms.EDProducer("TrackListMerger",
    Epsilon = cms.double(-0.001),
    FoundHitBonus = cms.double(5.0),
    LostHitPenalty = cms.double(5.0),
    MaxNormalizedChisq = cms.double(1000.0),
    MinFound = cms.int32(3),
    MinPT = cms.double(0.9),
    ShareFrac = cms.double(0.19),
    TrackProducers = cms.VInputTag("hltInitialStepTrackSelectionHighPurity", "hltHighPtTripletStepTrackSelectionHighPurity"),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(True),
    copyMVA = cms.bool(False),
    hasSelector = cms.vint32(0, 0),
    indivShareFrac = cms.vdouble(1.0, 1.0),
    makeReKeyedSeeds = cms.untracked.bool(False),
    newQuality = cms.string('confirmed'),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hltInitialStepTrackSelectionHighPurity"), cms.InputTag("hltHighPtTripletStepTrackSelectionHighPurity")),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0, 1)
    )),
    trackAlgoPriorityOrder = cms.string('trackAlgoPriorityOrder'),
    writeOnlyTrkQuals = cms.bool(False)
)

_hltGeneralTracksSingleIterPatatrack = hltGeneralTracks.clone(
    TrackProducers = ["hltInitialStepTrackSelectionHighPurity"],
    hasSelector = [0],
    indivShareFrac = [1.0],
    selectedTrackQuals = ["hltInitialStepTrackSelectionHighPurity"],
    setsToMerge = [cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0)
    )]
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
singleIterPatatrack.toReplaceWith(hltGeneralTracks, _hltGeneralTracksSingleIterPatatrack)

_hltGeneralTracksLST = hltGeneralTracks.clone(
    TrackProducers = ["hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTrackSelectionHighPuritypLSTCLST", "hltInitialStepTracksT5TCLST", "hltHighPtTripletStepTrackSelectionHighPurity"],
    hasSelector = [0,0,0,0],
    indivShareFrac = [0.1,0.1,0.1,0.1],
    selectedTrackQuals = ["hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTrackSelectionHighPuritypLSTCLST", "hltInitialStepTracksT5TCLST", "hltHighPtTripletStepTrackSelectionHighPurity"],
    setsToMerge = [cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0,1,2,3)
    )]
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(hltGeneralTracks, _hltGeneralTracksLST)

_hltGeneralTracksLSTSingleIterPatatrack = hltGeneralTracks.clone(
    TrackProducers = ["hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTrackSelectionHighPuritypLSTCLST", "hltInitialStepTracksT5TCLST"],
    hasSelector = [0,0,0],
    indivShareFrac = [0.1,0.1,0.1],
    selectedTrackQuals = ["hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTrackSelectionHighPuritypLSTCLST", "hltInitialStepTracksT5TCLST"],
    setsToMerge = [cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0,1,2)
    )]
)

(singleIterPatatrack & trackingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksLSTSingleIterPatatrack)

_hltGeneralTracksLSTSeeding = hltGeneralTracks.clone(
            TrackProducers = ["hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTracksT5TCLST", "hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST"],
            hasSelector = [0,0,0],
            indivShareFrac = [0.1,0.1,0.1],
            selectedTrackQuals = ["hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTracksT5TCLST", "hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST"],
            setsToMerge = [cms.PSet(
               pQual = cms.bool(True),
               tLists = cms.vint32(0,1,2)
            )]
    )

from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
(seedingLST & trackingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksLSTSeeding)
