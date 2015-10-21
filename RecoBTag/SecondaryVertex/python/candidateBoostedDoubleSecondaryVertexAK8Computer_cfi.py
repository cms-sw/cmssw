import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.trackSelection_cff import *

candidateBoostedDoubleSecondaryVertexAK8Computer = cms.ESProducer("CandidateBoostedDoubleSecondaryVertexESProducer",
    trackSelectionBlock,
    beta = cms.double(1.0),
    R0 = cms.double(0.8),
    maxSVDeltaRToJet = cms.double(0.7),
    useCondDB = cms.bool(False),
    weightFile = cms.FileInPath('RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT_v2.weights.xml.gz'),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(False),
    trackPairV0Filter = cms.PSet(k0sMassWindow = cms.double(0.03))
)

candidateBoostedDoubleSecondaryVertexAK8Computer.trackSelection.jetDeltaRMax = cms.double(0.8)
