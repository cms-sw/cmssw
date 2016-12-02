import FWCore.ParameterSet.Config as cms

deepFlavourJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('deepNNTagInfos'),
	NNConfig = cms.FileInPath('RecoBTag/DeepFlavour/data/DeepFlavourNoSL.json')
	)
	
deepFlavourCMVAJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('deepCMVATagInfos'),
	NNConfig = cms.FileInPath('RecoBTag/DeepFlavour/data/Model_DeepCMVA.json')
)
