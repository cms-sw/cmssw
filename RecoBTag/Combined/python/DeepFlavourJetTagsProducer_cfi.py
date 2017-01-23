import FWCore.ParameterSet.Config as cms

pfDeepCSVJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('pfDeepCSVTagInfos'),
	NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepFlavourNoSL.json')
	)
	
pfDeepCMVAJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('pfDeepCMVATagInfos'),
	NNConfig = cms.FileInPath('RecoBTag/Combined/data/Model_DeepCMVA.json')
)
