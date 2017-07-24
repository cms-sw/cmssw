import FWCore.ParameterSet.Config as cms

pfDeepFlavourTagInfos = cms.EDProducer(
	'DeepFlavourTagInfoProducer',
  jets = cms.InputTag("ak4PFJetsCHS"),
  vertices = cms.InputTag("offlinePrimaryVertices"),
  secondary_vertices = cms.InputTag("secondaryVertices"),
  shallow_tag_infos = cms.InputTag('pfDeepCSVTagInfos'),
  jet_radius = cms.double(0.4)
)
