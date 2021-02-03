import FWCore.ParameterSet.Config as cms

JPTZSPCorrectorAntiKt4 = cms.PSet(
    DzVertexCut = cms.double(0.2),
    EfficiencyMap = cms.string('CondFormats/JetMETObjects/data/CMSSW_538_TrackNonEff.txt'),
    ElectronIds = cms.InputTag("JPTeidTight"),
    Electrons = cms.InputTag("gedGsfElectrons"),
    JetSplitMerge = cms.int32(0),
    JetTracksAssociationAtCaloFace = cms.InputTag("ak4JetTracksAssociatorAtCaloFace"),
    JetTracksAssociationAtVertex = cms.InputTag("ak4JetTracksAssociatorAtVertexJPT"),
    LeakageMap = cms.string('CondFormats/JetMETObjects/data/CMSSW_538_TrackLeakage.txt'),
    MaxJetEta = cms.double(3.0),
    Muons = cms.InputTag("muons"),
    PtErrorQuality = cms.double(0.05),
    ResponseMap = cms.string('CondFormats/JetMETObjects/data/CMSSW_538_response.txt'),
    TrackQuality = cms.string('highPurity'),
    UseEfficiency = cms.bool(True),
    UseElectrons = cms.bool(True),
    UseInConeTracks = cms.bool(True),
    UseMuons = cms.bool(True),
    UseOutOfConeTracks = cms.bool(True),
    UseOutOfVertexTracks = cms.bool(True),
    UsePions = cms.bool(True),
    UseResponseInVecCorr = cms.bool(False),
    UseTrackQuality = cms.bool(True),
    VectorialCorrection = cms.bool(True),
    Verbose = cms.bool(True)
)