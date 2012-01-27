import FWCore.ParameterSet.Config as cms

vertexFromMuon = cms.EDProducer("VertexFromTrackProducer",
    verbose = cms.untracked.bool(False),
    trackLabel = cms.InputTag("hltL3MuonCandidates"),
    isRecoCandidate = cms.bool(True),
    useBeamSpot = cms.bool(True),
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
    useVertex = cms.bool(True),
    vertexLabel = cms.InputTag("hltPixelVertices"),
)

vertexFromElectron = cms.EDProducer("VertexFromTrackProducer",
    verbose = cms.untracked.bool(False),
    trackLabel = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    isRecoCandidate = cms.bool(True),
    useBeamSpot = cms.bool(True),
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
    useVertex = cms.bool(True),
    vertexLabel = cms.InputTag("hltPixelVertices"),
)

vertexFromTrack = cms.EDProducer("VertexFromTrackProducer",
    verbose = cms.untracked.bool(False),
    trackLabel = cms.InputTag("hltIter4Merged"),
    isRecoCandidate = cms.bool(False),
    useBeamSpot = cms.bool(True),
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
    useVertex = cms.bool(True),
    vertexLabel = cms.InputTag("hltPixelVertices"),
)
