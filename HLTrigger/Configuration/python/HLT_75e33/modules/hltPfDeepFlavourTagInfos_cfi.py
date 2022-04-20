import FWCore.ParameterSet.Config as cms

hltPfDeepFlavourTagInfos = cms.EDProducer("DeepFlavourTagInfoProducer",
    candidates = cms.InputTag("particleFlowTmp"),
    compute_probabilities = cms.bool(False),
    fallback_puppi_weight = cms.bool(False),
    fallback_vertex_association = cms.bool(False),
    flip = cms.bool(False),
    jet_radius = cms.double(0.4),
    jets = cms.InputTag("hltAK4PFPuppiJets"),
    max_jet_eta = cms.double(2.5),
    mightGet = cms.optional.untracked.vstring,
    min_candidate_pt = cms.double(0.95),
    min_jet_pt = cms.double(15),
    puppi_value_map = cms.InputTag("hltPFPuppi"),
    run_deepVertex = cms.bool(False),
    secondary_vertices = cms.InputTag("hltDeepInclusiveSecondaryVerticesPF"),
    shallow_tag_infos = cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfosPuppi"),
    vertex_associator = cms.InputTag("hltPrimaryVertexAssociation","original"),
    vertices = cms.InputTag("offlinePrimaryVertices")
)
