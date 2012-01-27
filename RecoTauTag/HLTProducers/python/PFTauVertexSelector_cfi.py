import FWCore.ParameterSet.Config as cms

pfTauVertexSelector = cms.EDFilter("PFTauVertexSelector",
    # Tau collection
    tauSrc = cms.InputTag('hltPFTaus'),

    # Vertex from primary vertex collection
    vertexSrc = cms.InputTag("hltPixelVertices"),
    
    # use leading track instead of primary vertex collection
    useLeadingTrack = cms.bool(False),
    
    # Vertex from leading track to be used
    trackSrc = cms.InputTag("hltIter4Merged"),
    
    # use leading RecoCandidate instead of primary vertex collection
    useLeadingRecoCandidate = cms.bool(False),
    
    # Vertex from RecoCandidate(e.g. lepton) track to be used
    recoCandidateSrc = cms.InputTag("hltL3MuonCandidates"),
    
    # max dZ distance to primary vertex
    dZ = cms.double(0.2),

    # filter events with at least N taus from PV
    filterOnNTaus = cms.uint32(0),
)
