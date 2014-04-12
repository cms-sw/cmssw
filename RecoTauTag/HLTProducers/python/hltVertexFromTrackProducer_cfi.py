import FWCore.ParameterSet.Config as cms

hltVertexFromTrackProducer = cms.EDProducer("VertexFromTrackProducer",
    verbose = cms.untracked.bool(False),

    # If isRecoCandidate=True "trackLabel" is used and assumed to be collection of candidates.
    # Otherwise it is assumed that "trackLabel" is collection of tracks and is used when useTriggerFilterElectrons=False and useTriggerFilterMuons=False
    isRecoCandidate = cms.bool(False),

    # Collection of tracks or candidates
    trackLabel = cms.InputTag("hltL3MuonCandidates"),
                                            
    # Use leading electron from TriggerObjectsWithRefs to determine z vertex position
    useTriggerFilterElectrons = cms.bool(False),

    # Electron TriggerObjectsWithRefs collection
    triggerFilterElectronsSrc = cms.InputTag("hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoL1JetTrackIsoFilter"),

    # Use leading muon from TriggerObjectsWithRefs to determine z vertex position
    useTriggerFilterMuons = cms.bool(True),                            

    # Muon TriggerObjectsWithRefs collection
    triggerFilterMuonsSrc = cms.InputTag("hltSingleMuIsoL3IsoFiltered15"),

    # Use beam spot for x/y vertex position
    useBeamSpot = cms.bool(True),
    # Beamspot collection
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
                                            
    # Use vertex for x/y vertex position (beam spot is used when PV does not exit)
    useVertex = cms.bool(True),
    # Vertex collection
    vertexLabel = cms.InputTag("hltPixelVertices"),
)

