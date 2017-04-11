import FWCore.ParameterSet.Config as cms

pfDeepCSVJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('pfDeepCSVTagInfos'),
  checkSVForDefaults = cms.bool(False),
  meanPadding = cms.bool(False),
	NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepFlavourNoSL.json')
	)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(pfDeepCSVJetTags, NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepCSV_PhaseI.json'))
phase1Pixel.toModify(pfDeepCSVJetTags, checkSVForDefaults = cms.bool(True))
phase1Pixel.toModify(pfDeepCSVJetTags, meanPadding = cms.bool(True))
