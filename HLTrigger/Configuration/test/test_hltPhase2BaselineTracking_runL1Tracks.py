import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTX")

### Load all ESSources, ESProducers and PSets
process.load("HLTrigger.Configuration.Phase2.hltPhase2Setup_cff")

### ES Prefers... would "prefer" to have them somewhere else
process.prefer("es_hardcode")
process.prefer("ppsDBESSource")
process.prefer("siPixelFakeGainOfflineESSource")

### GlobalTag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "111X_mcRun4_realistic_T15_v1"

### Rerun the module that makes TrackTrigger tracks
process.TTTracksFromTrackletEmulation = cms.EDProducer(
    "L1FPGATrackProducer",
    BeamSpotSource=cms.InputTag("offlineBeamSpot"),
    DTCLinkFile=cms.FileInPath(
        "L1Trigger/TrackFindingTracklet/data/calcNumDTCLinks.txt"
    ),
    DTCLinkLayerDiskFile=cms.FileInPath(
        "L1Trigger/TrackFindingTracklet/data/dtclinklayerdisk.dat"
    ),
    Extended=cms.bool(False),
    Hnpar=cms.uint32(4),
    MCTruthClusterInputTag=cms.InputTag(
        "TTClusterAssociatorFromPixelDigis", "ClusterAccepted"
    ),
    MCTruthStubInputTag=cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
    TTStubSource=cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),
    TrackingParticleInputTag=cms.InputTag("mix", "MergedTrackTruth"),
    TrackingVertexInputTag=cms.InputTag("mix", "MergedTrackTruth"),
    asciiFileName=cms.untracked.string(""),
    fitPatternFile=cms.FileInPath("L1Trigger/TrackFindingTracklet/data/fitpattern.txt"),
    memoryModulesFile=cms.FileInPath(
        "L1Trigger/TrackFindingTracklet/data/memorymodules_hourglass.dat"
    ),
    moduleCablingFile=cms.FileInPath(
        "L1Trigger/TrackFindingTracklet/data/modules_T5v3_27SP_nonant_tracklet.dat"
    ),
    processingModulesFile=cms.FileInPath(
        "L1Trigger/TrackFindingTracklet/data/processingmodules_hourglass.dat"
    ),
    readMoreMcTruth=cms.bool(True),
    wiresFile=cms.FileInPath("L1Trigger/TrackFindingTracklet/data/wires_hourglass.dat"),
)

process.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

process.hltPhase2GeneralTracks = cms.EDProducer(
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

process.load("HLTrigger.Configuration.Phase2.trackerLocalReco_cff")
process.load("HLTrigger.Configuration.Phase2.hltPhase2L1TracksSeqPattern_cff")
process.load("HLTrigger.Configuration.Phase2.hltPhase2PixelTracksSequenceL1_cff")
# Trimmed vertices are not used?
process.load("HLTrigger.Configuration.Phase2.hltPhase2PixelVerticesSequence_cff")
process.load("HLTrigger.Configuration.Phase2.hltPhase2InitialStepSequence_cff")
process.load("HLTrigger.Configuration.Phase2.hltPhase2HighPtTripletStepSequence_cff")

process.hltPhase2BaselineTrackingSeq = cms.Sequence(
    process.offlineBeamSpot
    + process.trackerLocalRecoSequence
    + process.TTTracksFromTrackletEmulation
    + process.hltPhase2L1TracksSeqPattern
    + process.hltPhase2PixelTracksSequenceL1
    + process.hltPhase2PixelVerticesSequence
    + process.hltPhase2InitialStepSequence
    + process.hltPhase2HighPtTripletStepSequence
    + process.hltPhase2GeneralTracks
)

process.HLT_Test_Path = cms.Path(process.hltPhase2BaselineTrackingSeq)

process.source = cms.Source(
    "PoolSource", fileNames=cms.untracked.vstring("file:local.root",),
)

process.maxEvents.input = cms.untracked.int32(100)
