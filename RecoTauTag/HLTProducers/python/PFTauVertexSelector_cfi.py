import FWCore.ParameterSet.Config as cms

pfTauVertexSelector = cms.EDFilter("PFTauVertexSelector",
    # Tau collection
    tauSrc = cms.InputTag('hltPFTaus'),

    # Use vertex collection for x/y vertex position
    useVertex = cms.bool(True),

    # Vertex collection
    vertexSrc = cms.InputTag("hltPixelVertices"),
    
    # Use beamspot as fallback for x/y vertex position
    useBeamSpot = cms.bool(True),

    # Beamspot collection
    beamSpotSrc = cms.InputTag("hltBeamSpot"),
    
    # use leading track to determine z vertex position
    useLeadingTrack = cms.bool(False),
    
    # Track collection
    trackSrc =cms.VInputTag(cms.InputTag("hltIter4Merged"),),
    
    # use leading RecoCandidate to determine z vertex position
    useLeadingRecoCandidate = cms.bool(False),
    
    # RecoCandidate(e.g. lepton) collection
    recoCandidateSrc = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"),),
    
    # use leading electron from TriggerObjectsWithRefs to determine z vertex position
    useTriggerFilterElectrons = cms.bool(False),
    
    # electron TriggerObjectsWithRefs collection
    triggerFilterElectronsSrc = cms.InputTag("hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoL1JetTrackIsoFilter"),
    
    # use leading muon from TriggerObjectsWithRefs to determine z vertex position
    useTriggerFilterMuons = cms.bool(False),
    
    # muon TriggerObjectsWithRefs collection
    triggerFilterMuonsSrc = cms.InputTag("hltSingleMuIsoL3IsoFiltered15"),
    
    # max dZ distance to primary vertex
    dZ = cms.double(0.2),

    # filter events with at least N taus from PV
    filterOnNTaus = cms.uint32(0),
)
