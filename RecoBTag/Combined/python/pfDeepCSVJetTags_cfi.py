import FWCore.ParameterSet.Config as cms

pfDeepCSVJetTags = cms.EDProducer(
	'DeepFlavourJetTagsProducer',
	src = cms.InputTag('pfDeepCSVTagInfos'),
  checkSVForDefaults = cms.bool(False),
  meanPadding = cms.bool(False),
	NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepFlavourNoSL.json'),
  toAdd = cms.PSet(
      probcc = cms.string('probc')
      ),
	)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(pfDeepCSVJetTags, NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepCSV_PhaseI.json'))
phase1Pixel.toModify(pfDeepCSVJetTags, checkSVForDefaults = cms.bool(True))
phase1Pixel.toModify(pfDeepCSVJetTags, meanPadding = cms.bool(True))
phase1Pixel.toModify(pfDeepCSVJetTags, toAdd = cms.PSet())

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(pfDeepCSVJetTags, NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepCSV_PhaseII.json'))
phase2_common.toModify(pfDeepCSVJetTags, checkSVForDefaults = cms.bool(True))
phase2_common.toModify(pfDeepCSVJetTags, meanPadding = cms.bool(True))
phase2_common.toModify(pfDeepCSVJetTags, toAdd = cms.PSet())
