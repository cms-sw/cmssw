import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.trackSelection_cff import *

candidateBoostedDoubleSecondaryVertexAK8Computer = cms.ESProducer("CandidateBoostedDoubleSecondaryVertexESProducer",
    useCondDB = cms.bool(False),
    weightFile = cms.FileInPath('RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT_v4.weights.xml.gz'),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(False)
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(candidateBoostedDoubleSecondaryVertexAK8Computer, weightFile = 'RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT_PhaseI_v1.weights.xml.gz')
