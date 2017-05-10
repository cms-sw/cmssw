import FWCore.ParameterSet.Config as cms

caloDeepCSVJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('caloDeepCSVTagInfos'),
  checkSVForDefaults = cms.bool(False),
  meanPadding = cms.bool(False),
	NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepFlavourNoSL.json'),
  toAdd = cms.PSet(
      probcc = cms.string('probc')
      ),
	)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(caloDeepCSVJetTags, NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepCSV_PhaseI.json'))
phase1Pixel.toModify(caloDeepCSVJetTags, checkSVForDefaults = cms.bool(True))
phase1Pixel.toModify(caloDeepCSVJetTags, meanPadding = cms.bool(True))
phase1Pixel.toModify(caloDeepCSVJetTags, toAdd = cms.PSet())

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(caloDeepCSVJetTags, NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepCSV_PhaseI.json'))
phase2_common.toModify(caloDeepCSVJetTags, checkSVForDefaults = cms.bool(True))
phase2_common.toModify(caloDeepCSVJetTags, meanPadding = cms.bool(True))
phase2_common.toModify(caloDeepCSVJetTags, toAdd = cms.PSet())
