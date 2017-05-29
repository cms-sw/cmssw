import FWCore.ParameterSet.Config as cms
	
pfDeepCMVAJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('pfDeepCMVATagInfos'),
	NNConfig = cms.FileInPath('RecoBTag/Combined/data/Model_DeepCMVA.json')
)
