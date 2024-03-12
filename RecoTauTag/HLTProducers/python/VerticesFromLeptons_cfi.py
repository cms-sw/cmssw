import FWCore.ParameterSet.Config as cms

vertexFromMuon = cms.EDProducer("VertexFromTrackProducer",
    verbose = cms.untracked.bool(False),
    trackLabel = cms.InputTag("hltL3MuonCandidates"),
    isRecoCandidate = cms.bool(False),
    useTriggerFilterElectrons = cms.bool(False),
    triggerFilterElectronsSrc = cms.InputTag(""),
    useTriggerFilterMuons = cms.bool(True),                            
    triggerFilterMuonsSrc = cms.InputTag("hltSingleMuIsoL3IsoFiltered15"),
    useBeamSpot = cms.bool(True),
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
    useVertex = cms.bool(True),
    vertexLabel = cms.InputTag("hltPixelVertices"),
)

vertexFromElectron = cms.EDProducer("VertexFromTrackProducer",
    verbose = cms.untracked.bool(False),
    trackLabel = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    isRecoCandidate = cms.bool(False),
    useTriggerFilterElectrons = cms.bool(True),
    triggerFilterElectronsSrc = cms.InputTag("hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoL1JetTrackIsoFilter"),
    useTriggerFilterMuons = cms.bool(False),                            
    triggerFilterMuonsSrc = cms.InputTag(""),
    useBeamSpot = cms.bool(True),
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
    useVertex = cms.bool(True),
    vertexLabel = cms.InputTag("hltPixelVertices"),
)

vertexFromTrack = cms.EDProducer("VertexFromTrackProducer",
    verbose = cms.untracked.bool(False),
    trackLabel = cms.InputTag("hltIter4Merged"),
    isRecoCandidate = cms.bool(False),
    useTriggerFilterElectrons = cms.bool(False),
    triggerFilterElectronsSrc = cms.InputTag(""),
    useTriggerFilterMuons = cms.bool(False),                            
    triggerFilterMuonsSrc = cms.InputTag(""),
    useBeamSpot = cms.bool(True),
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
    useVertex = cms.bool(True),
    vertexLabel = cms.InputTag("hltPixelVertices"),
)
# foo bar baz
# uW0wngJH1YGDh
# 5R3GgOzILLwoB
