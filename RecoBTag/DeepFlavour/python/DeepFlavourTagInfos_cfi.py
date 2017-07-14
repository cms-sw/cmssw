import FWCore.ParameterSet.Config as cms

pfDeepFlavourTagInfos = cms.EDProducer(
	'DeepFlavourTagInfoProducer',
  jets = cms.InputTag("slimmedJets"),
  secondary_vertices = cms.InputTag("slimmedSecondaryVertices"),
  shallow_tag_infos = cms.InputTag('pfDeepCSVTagInfos')
)
