import FWCore.ParameterSet.Config as cms

pfDeepFlavourTagInfos = cms.EDProducer(
	'DeepFlavourTagInfoProducer',
  jets = cms.InputTag("slimmedJets"),
  jets_uncorrected = cms.InputTag("slimmedJets"),
  vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
  secondary_vertices = cms.InputTag("slimmedSecondaryVertices"),
  shallow_tag_infos = cms.InputTag('pfDeepCSVTagInfos'),
  jet_radius = cms.double(0.4)
)
