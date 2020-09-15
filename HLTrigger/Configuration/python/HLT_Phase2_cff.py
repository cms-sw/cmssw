# Preliminary implementation of configuration filefragment (cff)
# For HLT Phase2

# Author: Thiago Tomei (SPRACE-Unesp)

import FWCore.ParameterSet.Config as cms

fragment = cms.ProcessFragment("HLT")

### Load all ESSources, ESProducers and PSets
fragment.load("HLTrigger.Configuration.Phase2.hltPhase2Setup_cff")

fragment.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

fragment.hltPhase2GeneralTracks = cms.EDProducer(
    "TrackListMerger",
    Epsilon=cms.double(-0.001),
    FoundHitBonus=cms.double(5.0),
    LostHitPenalty=cms.double(5.0),
    MaxNormalizedChisq=cms.double(1000.0),
    MinFound=cms.int32(3),
    MinPT=cms.double(0.9),
    ShareFrac=cms.double(0.19),
    TrackProducers=cms.VInputTag(
        "hltPhase2L1CtfTracks",
        "hltPhase2InitialStepTrackSelectionHighPurity",
        "hltPhase2HighPtTripletStepTrackSelectionHighPurity",
    ),
    allowFirstHitShare=cms.bool(True),
    copyExtras=cms.untracked.bool(True),
    copyMVA=cms.bool(False),
    hasSelector=cms.vint32(0, 0, 0),
    indivShareFrac=cms.vdouble(1.0, 1.0, 1.0),
    makeReKeyedSeeds=cms.untracked.bool(False),
    newQuality=cms.string("confirmed"),
    selectedTrackQuals=cms.VInputTag(
        cms.InputTag(""),
        cms.InputTag("hltPhase2InitialStepTrackSelectionHighPurity"),
        cms.InputTag("hltPhase2HighPtTripletStepTrackSelectionHighPurity"),
    ),
    setsToMerge=cms.VPSet(cms.PSet(pQual=cms.bool(True), tLists=cms.vint32(0, 1))),
    trackAlgoPriorityOrder=cms.string("trackAlgoPriorityOrder"),
    writeOnlyTrkQuals=cms.bool(False),
)

fragment.load("HLTrigger.Configuration.Phase2.hltPhase2TrackerLocalReco_cff")
fragment.load("HLTrigger.Configuration.Phase2.hltPhase2L1TracksSeqPattern_cff")
fragment.load("HLTrigger.Configuration.Phase2.hltPhase2PixelTracksSequenceL1_cff")
# Trimmed vertices are not used?
fragment.load("HLTrigger.Configuration.Phase2.hltPhase2PixelVerticesSequence_cff")
fragment.load("HLTrigger.Configuration.Phase2.hltPhase2InitialStepSequence_cff")
fragment.load("HLTrigger.Configuration.Phase2.hltPhase2HighPtTripletStepSequence_cff")

fragment.hltPhase2BaselineTrackingSeq = cms.Sequence(
    fragment.offlineBeamSpot
    + fragment.hltPhase2TrackerLocalRecoSequence
    + fragment.hltPhase2L1TracksSeqPattern
    + fragment.hltPhase2PixelTracksSequenceL1
    + fragment.hltPhase2PixelVerticesSequence
    + fragment.hltPhase2InitialStepSequence
    + fragment.hltPhase2HighPtTripletStepSequence
    + fragment.hltPhase2GeneralTracks
)

fragment.HLT_BaselineTracking = cms.Path(fragment.hltPhase2BaselineTrackingSeq)

fragment.HLTSchedule = cms.Schedule(*(fragment.HLT_BaselineTracking,))

### Suggestions from Silvio Donato's:
### Make sure that we always es_prefer our own ESModules instead
### of the ones that come from the vanilla configuration.
for esModule in fragment.es_producers_():
    if "hltPhase2" in esModule:
        setattr(
            fragment,
            "es_prefer_" + esModule,
            cms.ESPrefer(getattr(fragment, esModule).type_(), esModule),
        )
