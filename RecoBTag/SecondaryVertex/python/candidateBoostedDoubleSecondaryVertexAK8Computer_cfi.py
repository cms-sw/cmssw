import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.trackSelection_cff import *

candidateBoostedDoubleSecondaryVertexAK8Computer = cms.ESProducer("CandidateBoostedDoubleSecondaryVertexESProducer",
    useCondDB = cms.bool(False),
    weightFile = cms.FileInPath('RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT_v4.weights.xml.gz'),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(False)
)
