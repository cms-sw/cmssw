import FWCore.ParameterSet.Config as cms
	
pfDeepCMVAJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('pfDeepCMVATagInfos'),
  checkSVForDefaults = cms.bool(False),
  meanPadding = cms.bool(False),
	NNConfig = cms.FileInPath('RecoBTag/Combined/data/Model_DeepCMVA.json'),
  toAdd = cms.PSet(
      ),
)
