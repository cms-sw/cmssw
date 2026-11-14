import FWCore.ParameterSet.Config as cms

hltParticleTransformerAK4TagInfos = cms.EDProducer("HLTParticleTransformerAK4TagInfoProducer",
    candidates = cms.InputTag("hltParticleFlowTmp"),
    jet_radius = cms.double(0.4),
    jets = cms.InputTag("hltAK4PFPuppiJets"),
    mightGet = cms.optional.untracked.vstring,
    min_candidate_pt = cms.double(0.1),
    min_jet_pt = cms.double(5),
    max_jet_eta = cms.double(2.1),
    secondary_vertices = cms.InputTag("hltDeepInclusiveMergedVerticesPF"),
    vertex_associator = cms.InputTag("hltPrimaryVertexAssociation","original"),
    vertices = cms.InputTag("hltOfflinePrimaryVertices")
)
