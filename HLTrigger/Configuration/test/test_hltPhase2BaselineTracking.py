import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTX")

### Load all ESSources, ESProducers and PSets
process.load("HLTrigger.Configuration.Phase2.hltPhase2Setup_cff")

### ES Prefers... would "prefer" to have them somewhere else
process.prefer("es_hardcode")
process.prefer("ppsDBESSource")
process.prefer("siPixelFakeGainOfflineESSource")

### Don't rerun the module that makes TrackTrigger tracks
# process.TTTracksFromTrackletEmulation = cms.EDProducer

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
    + process.hltPhase2L1TracksSeqPattern
    + process.hltPhase2PixelTracksSequenceL1
    + process.hltPhase2PixelVerticesSequence
    + process.hltPhase2InitialStepSequence
    + process.hltPhase2HighPtTripletStepSequence
    + process.hltPhase2GeneralTracks
)

process.HLT_Test_Path = cms.Path(process.hltPhase2BaselineTrackingSeq)

process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(
        "/store/relval/CMSSW_11_1_0_patch1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW-RECO/110X_mcRun4_realistic_v3_2026D49PU200_raw1100_ProdType1-v1/10000/FCD54039-AAE9-A041-B0A9-BAEB0B625F31.root"
    ),
)

process.maxEvents.input = cms.untracked.int32(100)
