import FWCore.ParameterSet.Config as cms

pfDeepCSVJetTags = cms.EDProducer('DeepFlavourJetTagsProducer',
    src = cms.InputTag('pfDeepCSVTagInfos'),
    checkSVForDefaults = cms.bool(False),
    meanPadding = cms.bool(False),
    NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepFlavourNoSL.json'),
    toAdd = cms.PSet(
        probcc = cms.string('probc')
    ),
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(pfDeepCSVJetTags, 
                     NNConfig = 'RecoBTag/Combined/data/DeepCSV_PhaseI.json',
                     checkSVForDefaults = True,
                     meanPadding = True,
                     toAdd = cms.PSet()
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(pfDeepCSVJetTags, 
                       NNConfig = 'RecoBTag/Combined/data/DeepCSV_PhaseII.json',
                       checkSVForDefaults = True,
                       meanPadding = True,
                       toAdd = cms.PSet()
)
